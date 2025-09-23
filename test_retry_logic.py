"""
TEST SCRIPT FOR RETRY LOGIC
Run this to verify the retry decorators work correctly
"""

import time
import random
from datetime import datetime
from typing import Any
from functools import wraps

# ============================================================
# COPY OF RETRY LOGIC FROM optimization_solutions.py
# ============================================================

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retriable_exceptions: tuple = (Exception,)
):
    """
    Professional retry decorator with exponential backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Log success only if it was after retry attempts
                    if attempt > 0:
                        print(f"Operation succeeded after {attempt} retry attempts")
                    
                    return result
                    
                except retriable_exceptions as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Don't retry if it's a non-retriable error
                    if any(term in error_msg for term in ['unauthorized', 'forbidden', 'invalid key', 'quota exceeded']):
                        raise
                    
                    # Check if we should retry
                    if attempt == max_retries:
                        print(f"Operation failed after {max_retries} attempts")
                        raise
                    
                    # Calculate delay with optional jitter
                    if jitter:
                        actual_delay = delay * (0.5 + random.random())
                    else:
                        actual_delay = delay
                    
                    actual_delay = min(actual_delay, max_delay)
                    
                    # Only display message for rate limits
                    if 'rate limit' in error_msg or '429' in error_msg:
                        print(f"Rate limit encountered. Retrying in {actual_delay:.0f} seconds...")
                    
                    time.sleep(actual_delay)
                    
                    # Increase delay for next attempt
                    delay *= exponential_base
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

def retry_azure_openai(max_retries: int = 3):
    """Specialized retry decorator for Azure OpenAI calls"""
    return retry_with_exponential_backoff(
        max_retries=max_retries,
        initial_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
        retriable_exceptions=(Exception,)
    )

def retry_azure_search(max_retries: int = 3):
    """Specialized retry decorator for Azure Search calls"""
    return retry_with_exponential_backoff(
        max_retries=max_retries,
        initial_delay=0.5,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True,
        retriable_exceptions=(Exception,)
    )

# ============================================================
# TEST FUNCTIONS THAT SIMULATE DIFFERENT FAILURE SCENARIOS
# ============================================================

class TestMetrics:
    """Track test execution metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.attempts = 0
        self.timestamps = []
        self.errors = []
    
    def record_attempt(self, error=None):
        self.attempts += 1
        self.timestamps.append(datetime.now())
        if error:
            self.errors.append(str(error))

# Global metrics tracker
metrics = TestMetrics()

# ============================================================
# TEST SCENARIO 1: Rate Limit (Should Retry)
# ============================================================

@retry_azure_openai(max_retries=3)
def test_rate_limit_scenario(fail_times: int = 2):
    """Simulates rate limit errors that should trigger retries"""
    metrics.record_attempt()
    
    if metrics.attempts <= fail_times:
        error = Exception("Error 429: Rate limit exceeded")
        metrics.record_attempt(error)
        raise error
    
    return f"Success after {metrics.attempts} attempts"

# ============================================================
# TEST SCENARIO 2: Timeout (Should Retry)
# ============================================================

@retry_azure_search(max_retries=3)
def test_timeout_scenario(fail_times: int = 2):
    """Simulates timeout errors that should trigger retries"""
    metrics.record_attempt()
    
    if metrics.attempts <= fail_times:
        error = Exception("Request timeout after 30 seconds")
        metrics.record_attempt(error)
        raise error
    
    return f"Success after {metrics.attempts} attempts"

# ============================================================
# TEST SCENARIO 3: Unauthorized (Should NOT Retry)
# ============================================================

@retry_azure_openai(max_retries=3)
def test_unauthorized_scenario():
    """Simulates unauthorized error that should NOT trigger retries"""
    metrics.record_attempt()
    error = Exception("Error 401: Unauthorized access - invalid API key")
    metrics.record_attempt(error)
    raise error

# ============================================================
# TEST SCENARIO 4: Quota Exceeded (Should NOT Retry)
# ============================================================

@retry_azure_openai(max_retries=3)
def test_quota_exceeded_scenario():
    """Simulates quota exceeded error that should NOT trigger retries"""
    metrics.record_attempt()
    error = Exception("Error 403: Quota exceeded for this month")
    metrics.record_attempt(error)
    raise error

# ============================================================
# TEST SCENARIO 5: Generic Error (Should Retry)
# ============================================================

@retry_azure_search(max_retries=2)
def test_generic_error_scenario(fail_times: int = 1):
    """Simulates generic errors that should trigger retries"""
    metrics.record_attempt()
    
    if metrics.attempts <= fail_times:
        error = Exception("Connection reset by peer")
        metrics.record_attempt(error)
        raise error
    
    return f"Success after {metrics.attempts} attempts"

# ============================================================
# TEST SCENARIO 6: Always Fails (Should Exhaust Retries)
# ============================================================

@retry_azure_openai(max_retries=2)
def test_always_fails_scenario():
    """Simulates a function that always fails"""
    metrics.record_attempt()
    error = Exception("Service unavailable")
    metrics.record_attempt(error)
    raise error

# ============================================================
# TEST RUNNER
# ============================================================

def run_test(test_name: str, test_func, *args, **kwargs):
    """Run a single test and display results"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    metrics.reset()
    start_time = time.time()
    
    try:
        result = test_func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        print(f"RESULT: SUCCESS")
        print(f"Response: {result}")
        print(f"Total attempts: {metrics.attempts}")
        print(f"Total time: {elapsed:.2f} seconds")
        
        # Show retry delays if any
        if len(metrics.timestamps) > 1:
            print(f"Retry delays:")
            for i in range(1, len(metrics.timestamps)):
                delay = (metrics.timestamps[i] - metrics.timestamps[i-1]).total_seconds()
                print(f"  Attempt {i} -> {i+1}: {delay:.2f} seconds")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        
        print(f"RESULT: FAILED")
        print(f"Error: {str(e)}")
        print(f"Total attempts: {metrics.attempts}")
        print(f"Total time: {elapsed:.2f} seconds")
        
        # Show retry delays if any
        if len(metrics.timestamps) > 1:
            print(f"Retry delays:")
            for i in range(1, len(metrics.timestamps)):
                delay = (metrics.timestamps[i] - metrics.timestamps[i-1]).total_seconds()
                print(f"  Attempt {i} -> {i+1}: {delay:.2f} seconds")
        
        return False

def run_all_tests():
    """Run all test scenarios"""
    print("\n" + "="*60)
    print("RETRY LOGIC TEST SUITE")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Rate limit (should retry and succeed)
    results.append(("Rate Limit Recovery", run_test(
        "Rate Limit - Should retry and succeed",
        test_rate_limit_scenario,
        fail_times=2
    )))
    
    # Test 2: Timeout (should retry and succeed)
    results.append(("Timeout Recovery", run_test(
        "Timeout - Should retry with short delays",
        test_timeout_scenario,
        fail_times=2
    )))
    
    # Test 3: Unauthorized (should NOT retry)
    results.append(("Unauthorized No Retry", run_test(
        "Unauthorized - Should fail immediately without retry",
        test_unauthorized_scenario
    )))
    
    # Test 4: Quota exceeded (should NOT retry)
    results.append(("Quota Exceeded No Retry", run_test(
        "Quota Exceeded - Should fail immediately without retry",
        test_quota_exceeded_scenario
    )))
    
    # Test 5: Generic error (should retry and succeed)
    results.append(("Generic Error Recovery", run_test(
        "Generic Error - Should retry and succeed",
        test_generic_error_scenario,
        fail_times=1
    )))
    
    # Test 6: Always fails (should exhaust retries)
    results.append(("Exhaust Retries", run_test(
        "Always Fails - Should exhaust all retries",
        test_always_fails_scenario
    )))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Retry logic is working correctly.")
    else:
        print(f"\n{total - passed} tests failed. Review the retry logic.")

# ============================================================
# INTERACTIVE TEST FUNCTIONS
# ============================================================

def test_custom_scenario():
    """Interactive test where you can configure the failure pattern"""
    
    print("\n" + "="*60)
    print("CUSTOM RETRY TEST")
    print("="*60)
    
    # Get user input
    max_retries = int(input("Max retries (default 3): ") or "3")
    fail_times = int(input("How many times should it fail before succeeding? ") or "2")
    error_type = input("Error type (rate_limit/timeout/generic) [default: rate_limit]: ") or "rate_limit"
    
    # Create custom test function
    @retry_with_exponential_backoff(max_retries=max_retries)
    def custom_test():
        metrics.record_attempt()
        
        if metrics.attempts <= fail_times:
            if error_type == "rate_limit":
                error = Exception("Error 429: Rate limit exceeded")
            elif error_type == "timeout":
                error = Exception("Request timeout")
            else:
                error = Exception("Generic error")
            
            metrics.record_attempt(error)
            raise error
        
        return f"Success after {metrics.attempts} attempts"
    
    # Run the test
    run_test(
        f"Custom Test (fail {fail_times} times, {error_type} error)",
        custom_test
    )

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--custom":
        # Run interactive custom test
        test_custom_scenario()
    else:
        # Run all predefined tests
        run_all_tests()
        
        print("\n" + "="*60)
        print("To run a custom test, use: python test_retry_logic.py --custom")
        print("="*60)