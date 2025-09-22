"""
IMMEDIATE INTEGRATION GUIDE
Copy these exact changes into your Streamlit app RIGHT NOW
"""

import streamlit as st
import json
import time
from typing import Optional

# ============================================================
# STEP 1: FIX JSON CACHING (DO THIS FIRST - BIGGEST IMPACT!)
# ============================================================

# WRONG - This is probably what you have now:
# def load_json_data():
#     with open('/path/to/1gb_file.json', 'r') as f:
#         return json.load(f)

# CORRECT - Replace with this:
@st.cache_resource  # CRITICAL: Must be cache_resource NOT cache_data!
def load_json_data():
    """This loads ONCE and shares across ALL users - saves 149GB of memory!"""
    file_path = '/path/to/your/1gb_file.json'  # UPDATE THIS PATH
    
    print(f"[{time.strftime('%H:%M:%S')}] Loading JSON file (this should only happen ONCE)...")
    start = time.time()
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"[{time.strftime('%H:%M:%S')}] JSON loaded in {time.time()-start:.2f} seconds")
    return data

# Use it in your app like this:
def your_main_app():
    # This will be instant after first load!
    shared_data = load_json_data()
    
    # Now use shared_data normally
    # process_data(shared_data)

# ============================================================
# STEP 2: ADD RETRY LOGIC TO AZURE OPENAI CALLS
# ============================================================

def call_azure_openai_with_retry(
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    max_retries: int = 3
):
    """
    Replace your current OpenAI calls with this function
    It handles rate limits, timeouts, and failures automatically
    """
    
    import time
    from openai import AzureOpenAI  # or whatever import you use
    
    # Your existing client initialization
    # client = AzureOpenAI(
    #     api_key=your_api_key,
    #     api_version=your_api_version,
    #     azure_endpoint=your_endpoint
    # )
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Show retry status in UI
            if attempt > 0:
                st.info(f"🔄 Retry attempt {attempt + 1}/{max_retries}...")
            
            # ADD YOUR ACTUAL AZURE OPENAI CALL HERE
            # Example:
            # response = client.chat.completions.create(
            #     model="your-deployment-name",
            #     messages=[{"role": "user", "content": prompt}],
            #     max_tokens=max_tokens,
            #     temperature=temperature,
            #     timeout=30  # Add 30 second timeout
            # )
            # return response.choices[0].message.content
            
            # For testing, simulate the call:
            if attempt < 1:  # Simulate failure on first attempt
                raise Exception("Simulated rate limit error")
            
            return f"Response to: {prompt[:50]}..."
            
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Check error type
            if 'rate limit' in error_msg or 'too many requests' in error_msg:
                # Rate limit - wait longer
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                st.warning(f"⏳ Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
            elif 'timeout' in error_msg:
                # Timeout - retry quickly
                st.warning(f"⏱️ Request timed out. Retrying...")
                time.sleep(1)
                
            elif 'quota' in error_msg:
                # Quota exceeded - don't retry
                st.error("❌ Azure OpenAI quota exceeded. Contact admin.")
                raise
                
            else:
                # Unknown error - retry with short delay
                time.sleep(1)
            
            # If this was the last retry, raise the error
            if attempt == max_retries - 1:
                st.error(f"❌ Failed after {max_retries} attempts: {str(last_error)}")
                raise last_error
    
    return None

# ============================================================
# STEP 3: ADD RETRY LOGIC TO AZURE AI SEARCH
# ============================================================

def search_azure_with_retry(
    query: str,
    top: int = 10,
    max_retries: int = 3
):
    """
    Replace your current Azure Search calls with this function
    It handles throttling and failures automatically
    """
    
    from azure.search.documents import SearchClient  # or whatever import you use
    from azure.core.credentials import AzureKeyCredential
    
    # Your existing client initialization
    # search_client = SearchClient(
    #     endpoint=your_search_endpoint,
    #     index_name=your_index_name,
    #     credential=AzureKeyCredential(your_search_key)
    # )
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Show retry status
            if attempt > 0:
                st.info(f"🔄 Search retry {attempt + 1}/{max_retries}...")
            
            # ADD YOUR ACTUAL AZURE SEARCH CALL HERE
            # Example:
            # results = search_client.search(
            #     search_text=query,
            #     top=top,
            #     include_total_count=True
            # )
            # return list(results)
            
            # For testing, simulate the call:
            if attempt < 1:  # Simulate failure
                raise Exception("Simulated throttling error")
            
            return [{"content": f"Result for: {query}"}]
            
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            if 'throttl' in error_msg or 'rate' in error_msg:
                # Throttled - wait with backoff
                wait_time = 1 * (attempt + 1)  # Linear backoff: 1, 2, 3 seconds
                st.warning(f"⏳ Search throttled. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
            elif 'timeout' in error_msg:
                # Timeout - retry quickly
                time.sleep(0.5)
                
            else:
                # Other error - short delay
                time.sleep(1)
            
            if attempt == max_retries - 1:
                st.error(f"❌ Search failed after {max_retries} attempts")
                # Return empty results instead of crashing
                return []
    
    return []

# ============================================================
# STEP 4: ADD CONCURRENT REQUEST LIMITING
# ============================================================

from threading import Semaphore

# Create these at the top of your app file
# These limit concurrent operations to prevent overload
llm_semaphore = Semaphore(10)  # Max 10 concurrent LLM calls
search_semaphore = Semaphore(20)  # Max 20 concurrent searches
pdf_semaphore = Semaphore(5)  # Max 5 concurrent PDF processing

def process_with_concurrency_limit(semaphore, func, *args, **kwargs):
    """
    Wrapper to limit concurrent operations
    """
    with semaphore:
        return func(*args, **kwargs)

# ============================================================
# STEP 5: COMPLETE EXAMPLE - YOUR APP FLOW
# ============================================================

def your_complete_app_flow():
    """
    This shows how to integrate everything into your existing app
    """
    
    st.title("Your App with Fixes")
    
    # 1. Load JSON data (cached and shared!)
    try:
        data = load_json_data()  # This is now efficient!
        st.success(f"✅ Data loaded: {len(data)} items")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return
    
    # 2. User input
    user_query = st.text_input("Enter your query")
    
    if st.button("Process"):
        # Progress tracking
        progress = st.progress(0)
        status = st.empty()
        
        # 3. Search with retry and concurrency limit
        status.text("🔍 Searching...")
        progress.progress(25)
        
        search_results = process_with_concurrency_limit(
            search_semaphore,
            search_azure_with_retry,
            user_query
        )
        
        # 4. Process results (your existing logic)
        status.text("📊 Processing results...")
        progress.progress(50)
        
        # Create prompt from search results
        context = "\n".join([str(r) for r in search_results[:5]])
        prompt = f"Context: {context}\n\nQuery: {user_query}\n\nResponse:"
        
        # 5. Call LLM with retry and concurrency limit
        status.text("🤖 Generating response...")
        progress.progress(75)
        
        response = process_with_concurrency_limit(
            llm_semaphore,
            call_azure_openai_with_retry,
            prompt,
            max_tokens=1000
        )
        
        # 6. Display results
        progress.progress(100)
        status.text("✅ Complete!")
        
        st.write("### Response")
        st.write(response)

# ============================================================
# STEP 6: QUICK TEST TO VERIFY FIXES
# ============================================================

def test_your_fixes():
    """
    Run this to verify your fixes are working
    """
    st.header("Testing Fixes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test JSON Cache"):
            start = time.time()
            data = load_json_data()
            load_time = time.time() - start
            
            if load_time < 0.1:
                st.success(f"✅ Cached! Load time: {load_time:.4f}s")
            else:
                st.warning(f"⚠️ Not cached? Load time: {load_time:.2f}s")
    
    with col2:
        if st.button("Test OpenAI Retry"):
            try:
                response = call_azure_openai_with_retry("Test prompt")
                st.success("✅ OpenAI retry working!")
            except Exception as e:
                st.error(f"❌ OpenAI retry failed: {e}")
    
    with col3:
        if st.button("Test Search Retry"):
            try:
                results = search_azure_with_retry("Test query")
                st.success(f"✅ Search retry working! {len(results)} results")
            except Exception as e:
                st.error(f"❌ Search retry failed: {e}")

# ============================================================
# MAIN APP ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Add this to your main app
    st.set_page_config(page_title="Your App", layout="wide")
    
    # Initialize (do this once at startup)
    st.sidebar.header("System Status")
    
    # Memory monitor
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    st.sidebar.metric("Memory Usage", f"{memory_mb:.0f} MB")
    
    # Add test controls
    with st.sidebar:
        if st.checkbox("Show Test Panel"):
            test_your_fixes()
    
    # Run main app
    your_complete_app_flow()

# ============================================================
# CRITICAL REMINDERS
# ============================================================

"""
CHECKLIST - DO THESE NOW:

✅ 1. Replace JSON loading with @st.cache_resource version
   - This is the MOST IMPORTANT fix
   - Will reduce memory from 150GB to 1GB

✅ 2. Wrap Azure OpenAI calls with retry logic
   - Prevents failures from rate limits
   - Adds timeout protection

✅ 3. Wrap Azure Search calls with retry logic  
   - Handles throttling gracefully
   - Returns empty results instead of crashing

✅ 4. Add semaphores to limit concurrent operations
   - Prevents overwhelming the services
   - Keeps app responsive

✅ 5. Test with the test buttons to verify everything works

TIME NEEDED: 15-20 minutes to implement all fixes

BIGGEST IMPACT: The JSON caching fix (#1) - do this first!
"""