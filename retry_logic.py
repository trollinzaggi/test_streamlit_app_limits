"""
Retry Logic Implementation for Azure OpenAI and Azure AI Search
Handles rate limits, timeouts, and transient failures gracefully
"""

import streamlit as st
import time
import random
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
from threading import Lock, Semaphore
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================
# RETRY STRATEGIES
# ================================================

class RetryStrategy(Enum):
    """Different retry strategies for different failure types"""
    EXPONENTIAL_BACKOFF = "exponential"
    LINEAR_BACKOFF = "linear"
    FIXED_DELAY = "fixed"
    JITTERED = "jittered"

class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        timeout: float = 30.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.timeout = timeout
        self.strategy = strategy

# Default configurations for different services
AZURE_OPENAI_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    timeout=30.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
)

AZURE_SEARCH_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=0.5,
    max_delay=30.0,
    timeout=15.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
)

# ================================================
# BASIC RETRY DECORATOR
# ================================================

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retriable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Basic retry decorator with exponential backoff
    
    Usage:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def call_azure_openai(prompt):
            return client.completions.create(prompt=prompt)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Log attempt
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt}/{max_retries} for {func.__name__}")
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Success - reset any error states if needed
                    if attempt > 0:
                        logger.info(f"Successfully completed {func.__name__} after {attempt} retries")
                    
                    return result
                    
                except retriable_exceptions as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {str(e)}")
                        raise
                    
                    # Calculate delay
                    if jitter:
                        actual_delay = delay * (0.5 + random.random())
                    else:
                        actual_delay = delay
                    
                    actual_delay = min(actual_delay, max_delay)
                    
                    # Call on_retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, actual_delay, e)
                    
                    # Log and wait
                    logger.warning(f"Error in {func.__name__}: {str(e)}. Retrying in {actual_delay:.2f} seconds...")
                    time.sleep(actual_delay)
                    
                    # Increase delay for next attempt
                    delay *= exponential_base
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

# ================================================
# ADVANCED RETRY WITH CIRCUIT BREAKER
# ================================================

class CircuitBreaker:
    """
    Circuit breaker pattern to prevent overwhelming failed services
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = Lock()
    
    def call(self, func, *args, **kwargs):
        with self._lock:
            # Check if circuit should be opened
            if self.state == 'open':
                if self.last_failure_time and \
                   (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = 'half-open'
                    logger.info(f"Circuit breaker entering half-open state for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}. Service is unavailable.")
            
            try:
                # Attempt the call
                result = func(*args, **kwargs)
                
                # Success - reset circuit if needed
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
                    logger.info(f"Circuit breaker CLOSED for {func.__name__}. Service recovered.")
                
                return result
                
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    logger.error(f"Circuit breaker OPEN for {func.__name__}. Too many failures ({self.failure_count})")
                
                raise

# ================================================
# AZURE OPENAI SPECIFIC RETRY IMPLEMENTATION
# ================================================

class AzureOpenAIRetryClient:
    """
    Wrapper for Azure OpenAI client with built-in retry logic
    """
    
    def __init__(self, client, config: RetryConfig = AZURE_OPENAI_CONFIG):
        self.client = client
        self.config = config
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.rate_limiter = RateLimiter(max_calls=50, window_seconds=60)
        self.request_semaphore = Semaphore(10)  # Max 10 concurrent requests
        
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def completion_with_retry(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Call Azure OpenAI with automatic retry on failure
        """
        # Check rate limit
        self.rate_limiter.acquire()
        
        # Limit concurrent requests
        with self.request_semaphore:
            try:
                # Add timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._make_completion_call,
                        prompt, max_tokens, temperature, **kwargs
                    )
                    result = future.result(timeout=self.config.timeout)
                    return result
                    
            except TimeoutError:
                logger.error(f"Azure OpenAI call timed out after {self.config.timeout}s")
                raise Exception(f"Request timed out after {self.config.timeout} seconds")
            
            except Exception as e:
                # Check for specific Azure OpenAI errors
                error_message = str(e).lower()
                
                if 'rate limit' in error_message:
                    # Rate limit hit - use longer backoff
                    logger.warning("Rate limit hit. Applying extended backoff...")
                    time.sleep(10)
                    raise
                
                elif 'quota exceeded' in error_message:
                    # Quota exceeded - don't retry
                    logger.error("Quota exceeded. Not retrying.")
                    st.error("⚠️ Azure OpenAI quota exceeded. Please contact admin.")
                    raise
                
                elif 'unauthorized' in error_message or 'forbidden' in error_message:
                    # Auth error - don't retry
                    logger.error("Authentication error. Not retrying.")
                    st.error("❌ Authentication failed. Please check credentials.")
                    raise
                
                else:
                    # Other errors - retry
                    raise
    
    def _make_completion_call(self, prompt, max_tokens, temperature, **kwargs):
        """Actual API call to Azure OpenAI"""
        return self.circuit_breaker.call(
            lambda: self.client.completions.create(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        )

# ================================================
# AZURE AI SEARCH SPECIFIC RETRY IMPLEMENTATION  
# ================================================

class AzureSearchRetryClient:
    """
    Wrapper for Azure AI Search client with built-in retry logic
    """
    
    def __init__(self, search_client, config: RetryConfig = AZURE_SEARCH_CONFIG):
        self.client = search_client
        self.config = config
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        self.cache = SearchCache(max_size=100, ttl_seconds=300)  # Cache for 5 minutes
    
    @retry_with_backoff(max_retries=3, initial_delay=0.5, max_delay=10.0)
    def search_with_retry(
        self,
        search_text: str,
        filter: Optional[str] = None,
        top: int = 10,
        **kwargs
    ):
        """
        Search Azure AI Search with automatic retry on failure
        """
        # Check cache first
        cache_key = f"{search_text}:{filter}:{top}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached search results for: {search_text}")
            return cached_result
        
        try:
            # Add timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._make_search_call,
                    search_text, filter, top, **kwargs
                )
                results = future.result(timeout=self.config.timeout)
                
                # Cache the results
                self.cache.set(cache_key, results)
                
                return results
                
        except TimeoutError:
            logger.error(f"Azure Search timed out after {self.config.timeout}s")
            
            # Return degraded response
            return {
                'results': [],
                'error': 'Search timeout - service may be overloaded',
                'degraded': True
            }
        
        except Exception as e:
            error_message = str(e).lower()
            
            if 'throttl' in error_message or 'rate' in error_message:
                # Throttling - wait longer
                time.sleep(5)
                raise
            
            elif 'index not found' in error_message:
                # Configuration error - don't retry
                logger.error("Search index not found. Check configuration.")
                st.error("❌ Search index configuration error.")
                raise
            
            else:
                # Other errors - retry
                raise
    
    def _make_search_call(self, search_text, filter, top, **kwargs):
        """Actual API call to Azure Search"""
        return self.circuit_breaker.call(
            lambda: self.client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                **kwargs
            )
        )

# ================================================
# RATE LIMITER
# ================================================

class RateLimiter:
    """
    Rate limiter to prevent overwhelming APIs
    """
    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = deque()
        self._lock = Lock()
    
    def acquire(self, block: bool = True):
        """
        Acquire permission to make a call
        """
        while True:
            with self._lock:
                now = time.time()
                
                # Remove old calls outside window
                while self.calls and self.calls[0] < now - self.window_seconds:
                    self.calls.popleft()
                
                # Check if we can make a call
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return True
                
                if not block:
                    return False
            
            # Wait before trying again
            time.sleep(0.1)

# ================================================
# SEARCH CACHE
# ================================================

class SearchCache:
    """Simple cache for search results"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self._lock = Lock()
    
    def get(self, key: str):
        with self._lock:
            if key in self.cache:
                # Check if expired
                if time.time() - self.access_times[key] > self.ttl_seconds:
                    del self.cache[key]
                    del self.access_times[key]
                    return None
                
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        with self._lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()

# ================================================
# STREAMLIT INTEGRATION EXAMPLES
# ================================================

def create_retry_wrapped_clients():
    """
    Example of how to create retry-wrapped clients in your Streamlit app
    """
    
    # Your existing Azure OpenAI client
    from azure.identity import DefaultAzureCredential
    # from azure.ai.openai import OpenAIClient  # Example import
    
    # Wrap with retry logic
    # original_openai_client = OpenAIClient(...)  # Your existing client
    # retry_openai_client = AzureOpenAIRetryClient(original_openai_client)
    
    # Your existing Azure Search client
    # from azure.search.documents import SearchClient  # Example import
    
    # Wrap with retry logic
    # original_search_client = SearchClient(...)  # Your existing client
    # retry_search_client = AzureSearchRetryClient(original_search_client)
    
    return None, None  # Return your wrapped clients

# ================================================
# READY-TO-USE FUNCTIONS FOR YOUR APP
# ================================================

@st.cache_resource
def get_openai_client_with_retry():
    """
    Singleton OpenAI client with retry logic
    Add this to your app to get a shared, retry-enabled client
    """
    # Import your actual Azure OpenAI setup here
    # from your_openai_module import create_openai_client
    
    # For now, returning a mock implementation
    class MockOpenAIClient:
        def completions_create(self, **kwargs):
            # Your actual implementation
            pass
    
    base_client = MockOpenAIClient()
    return AzureOpenAIRetryClient(base_client)

@st.cache_resource
def get_search_client_with_retry():
    """
    Singleton Search client with retry logic
    Add this to your app to get a shared, retry-enabled client
    """
    # Import your actual Azure Search setup here
    # from your_search_module import create_search_client
    
    # For now, returning a mock implementation
    class MockSearchClient:
        def search(self, **kwargs):
            # Your actual implementation
            pass
    
    base_client = MockSearchClient()
    return AzureSearchRetryClient(base_client)

# ================================================
# SIMPLE DROP-IN REPLACEMENTS
# ================================================

def call_openai_with_retry(prompt: str, max_tokens: int = 1000, **kwargs):
    """
    Direct replacement for your OpenAI calls
    Just replace your_client.complete(prompt) with this function
    """
    
    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        retriable_exceptions=(Exception,)
    )
    def _make_call():
        # ADD YOUR ACTUAL OPENAI CALL HERE
        # Example:
        # return your_openai_client.completions.create(
        #     prompt=prompt,
        #     max_tokens=max_tokens,
        #     **kwargs
        # )
        pass
    
    try:
        with st.spinner("Calling Azure OpenAI..."):
            result = _make_call()
            return result
    except Exception as e:
        st.error(f"Failed to get response from Azure OpenAI: {str(e)}")
        # Return a fallback response
        return {"error": str(e), "fallback": True}

def search_with_retry(query: str, top: int = 10, **kwargs):
    """
    Direct replacement for your Azure Search calls
    Just replace your_client.search(query) with this function
    """
    
    @retry_with_backoff(
        max_retries=3,
        initial_delay=0.5,
        max_delay=10.0,
        retriable_exceptions=(Exception,)
    )
    def _make_search():
        # ADD YOUR ACTUAL SEARCH CALL HERE
        # Example:
        # return your_search_client.search(
        #     search_text=query,
        #     top=top,
        #     **kwargs
        # )
        pass
    
    try:
        with st.spinner("Searching..."):
            results = _make_search()
            return results
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        # Return empty results as fallback
        return {"value": [], "error": str(e)}

# ================================================
# MONITORING AND METRICS
# ================================================

@st.cache_resource
def get_retry_metrics():
    """Track retry statistics"""
    return {
        'total_calls': 0,
        'successful_calls': 0,
        'failed_calls': 0,
        'retried_calls': 0,
        'total_retries': 0,
        'timeouts': 0,
        'circuit_breaker_trips': 0
    }

def display_retry_metrics():
    """Display retry metrics in Streamlit sidebar"""
    metrics = get_retry_metrics()
    
    with st.sidebar:
        st.subheader("🔄 Retry Metrics")
        
        success_rate = 0
        if metrics['total_calls'] > 0:
            success_rate = (metrics['successful_calls'] / metrics['total_calls']) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Success Rate", f"{success_rate:.1f}%")
            st.metric("Total Retries", metrics['total_retries'])
        
        with col2:
            st.metric("Timeouts", metrics['timeouts'])
            st.metric("Circuit Trips", metrics['circuit_breaker_trips'])

if __name__ == "__main__":
    # Test the retry logic
    st.title("Retry Logic Test")
    
    # Test retry decorator
    @retry_with_backoff(max_retries=3, initial_delay=0.5)
    def test_function(should_fail_times=0):
        """Test function that fails a specified number of times"""
        if not hasattr(test_function, 'call_count'):
            test_function.call_count = 0
        
        test_function.call_count += 1
        
        if test_function.call_count <= should_fail_times:
            raise Exception(f"Simulated failure {test_function.call_count}")
        
        return f"Success after {test_function.call_count} attempts"
    
    if st.button("Test Retry Logic"):
        try:
            result = test_function(should_fail_times=2)
            st.success(result)
        except Exception as e:
            st.error(f"Failed after retries: {e}")
    
    display_retry_metrics()