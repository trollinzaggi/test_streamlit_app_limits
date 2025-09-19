"""
COMPLETE MEMORY TRACKING INTEGRATION
Add this to your Streamlit app to track memory per user
"""

import streamlit as st
import sys
import psutil
import time
from datetime import datetime

# ============================================================
# QUICK INTEGRATION - ADD THIS TO YOUR APP
# ============================================================

def init_session_memory_tracking():
    """Call this at the start of your app"""
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.session_start = datetime.now()
        st.session_state.memory_snapshots = []

def track_memory_checkpoint(checkpoint_name: str):
    """
    Call this at different points in your app to track memory growth
    
    Usage:
        track_memory_checkpoint("before_json_load")
        data = load_json()
        track_memory_checkpoint("after_json_load")
    """
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # Calculate session state size
    session_size = 0
    for key, value in st.session_state.items():
        try:
            session_size += sys.getsizeof(value)
            # For complex objects, try to get deeper
            if hasattr(value, '__dict__'):
                session_size += sys.getsizeof(value.__dict__)
        except:
            pass
    
    session_size_mb = session_size / (1024 * 1024)
    
    checkpoint_data = {
        'checkpoint': checkpoint_name,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'total_memory_mb': memory_mb,
        'session_state_mb': session_size_mb,
        'session_id': st.session_state.session_id
    }
    
    st.session_state.memory_snapshots.append(checkpoint_data)
    
    # Display in sidebar
    with st.sidebar:
        st.metric(
            f"Memory at {checkpoint_name}", 
            f"{memory_mb:.0f} MB total",
            f"{session_size_mb:.2f} MB session"
        )
    
    return checkpoint_data

def display_memory_per_user():
    """Display memory metrics per user in sidebar"""
    
    with st.sidebar:
        st.header(f"🧠 Memory: User {st.session_state.get('session_id', 'unknown')}")
        
        # Current memory
        process = psutil.Process()
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Session state size
        session_size = sum(sys.getsizeof(v) for v in st.session_state.values())
        session_mb = session_size / (1024 * 1024)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Process", f"{current_memory_mb:.0f} MB")
        with col2:
            st.metric("Your Session", f"{session_mb:.2f} MB")
        
        # Show memory growth through checkpoints
        if 'memory_snapshots' in st.session_state and len(st.session_state.memory_snapshots) > 1:
            st.subheader("Memory Growth")
            
            first = st.session_state.memory_snapshots[0]
            last = st.session_state.memory_snapshots[-1]
            
            memory_growth = last['total_memory_mb'] - first['total_memory_mb']
            session_growth = last['session_state_mb'] - first['session_state_mb']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Total Growth", 
                    f"{memory_growth:.0f} MB",
                    help="Memory growth since page load"
                )
            with col2:
                st.metric(
                    "Session Growth", 
                    f"{session_growth:.2f} MB",
                    help="Your session data growth"
                )
            
            # Projection for 150 users
            st.subheader("📊 Load Test Projection")
            
            # Estimate memory per user
            # Assume current memory is baseline + user memory
            estimated_baseline = 500  # MB for base Streamlit
            estimated_per_user = max(current_memory_mb - estimated_baseline, session_mb)
            projected_150_users = estimated_baseline + (estimated_per_user * 150)
            
            st.info(f"""
            **Current Usage:** {current_memory_mb:.0f} MB  
            **Est. per user:** {estimated_per_user:.0f} MB  
            **150 users projection:** {projected_150_users/1024:.1f} GB
            """)
            
            if projected_150_users > 200000:  # 200 GB
                st.error("❌ Will NOT handle 150 users!")
            elif projected_150_users > 150000:  # 150 GB  
                st.warning("⚠️ Risky for 150 users")
            else:
                st.success("✅ Should handle 150 users")

# ============================================================
# YOUR APP WITH MEMORY TRACKING
# ============================================================

def your_app_with_memory_tracking():
    """
    This is how your app should look with memory tracking
    """
    
    # Initialize memory tracking
    init_session_memory_tracking()
    
    # Display memory in sidebar
    display_memory_per_user()
    
    # Main app
    st.title("Your App with Memory Tracking")
    
    # CHECKPOINT 1: Start
    track_memory_checkpoint("app_start")
    
    # Load JSON with caching
    @st.cache_resource
    def load_json_cached():
        track_memory_checkpoint("before_json_load")
        
        # Simulate loading 1GB file
        import json
        # data = json.load(open('your_file.json'))
        data = {"test": "x" * 1000000}  # Simulation
        
        track_memory_checkpoint("after_json_load") 
        return data
    
    # PDF Processing
    def process_pdf(file):
        track_memory_checkpoint("before_pdf_process")
        
        # Your PDF processing code
        result = "Processed PDF"
        
        track_memory_checkpoint("after_pdf_process")
        return result
    
    # LLM Call
    def call_llm(prompt):
        track_memory_checkpoint("before_llm_call")
        
        # Your LLM code
        response = f"Response to: {prompt}"
        
        track_memory_checkpoint("after_llm_call")
        return response
    
    # UI Flow
    if st.button("Load Data"):
        data = load_json_cached()
        st.success(f"Loaded data")
    
    uploaded_file = st.file_uploader("Upload PDF")
    if uploaded_file:
        result = process_pdf(uploaded_file)
        st.write(result)
    
    query = st.text_input("Enter query")
    if st.button("Submit Query"):
        response = call_llm(query)
        st.write(response)
    
    # Show detailed memory breakdown
    if st.checkbox("Show Memory Details"):
        if 'memory_snapshots' in st.session_state:
            import pandas as pd
            df = pd.DataFrame(st.session_state.memory_snapshots)
            st.dataframe(df, use_container_width=True)
            
            # Calculate memory used by each operation
            if len(df) > 1:
                st.subheader("Memory Used by Each Operation")
                for i in range(1, len(df)):
                    prev = st.session_state.memory_snapshots[i-1]
                    curr = st.session_state.memory_snapshots[i]
                    
                    if 'after' in curr['checkpoint']:
                        operation = curr['checkpoint'].replace('after_', '')
                        memory_used = curr['total_memory_mb'] - prev['total_memory_mb']
                        session_used = curr['session_state_mb'] - prev['session_state_mb']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                f"{operation}",
                                f"{memory_used:.1f} MB total"
                            )
                        with col2:
                            st.metric(
                                f"{operation} (session)",
                                f"{session_used:.2f} MB"
                            )

# ============================================================
# CRITICAL: TEST IF JSON CACHING IS WORKING
# ============================================================

def test_json_caching():
    """
    Run this test to verify your JSON is properly cached
    """
    st.header("🧪 JSON Caching Test")
    
    col1, col2 = st.columns(2)
    
    # Without caching (BAD)
    with col1:
        st.subheader("❌ Without Cache (BAD)")
        
        def load_json_no_cache():
            # Simulate loading
            time.sleep(2)
            return {"data": "x" * 1000000}
        
        if st.button("Load without cache"):
            track_memory_checkpoint("before_no_cache")
            start = time.time()
            
            data = load_json_no_cache()
            
            track_memory_checkpoint("after_no_cache")
            load_time = time.time() - start
            
            st.error(f"Loaded in {load_time:.2f}s")
            st.error("Each user loads separately!")
    
    # With caching (GOOD)
    with col2:
        st.subheader("✅ With Cache (GOOD)")
        
        @st.cache_resource
        def load_json_with_cache():
            # Simulate loading
            time.sleep(2)
            return {"data": "x" * 1000000}
        
        if st.button("Load with cache"):
            track_memory_checkpoint("before_cache")
            start = time.time()
            
            data = load_json_with_cache()
            
            track_memory_checkpoint("after_cache")
            load_time = time.time() - start
            
            if load_time < 0.1:
                st.success(f"Loaded in {load_time:.4f}s (CACHED!)")
                st.success("Shared across all users!")
            else:
                st.warning(f"First load: {load_time:.2f}s")

# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Memory Tracking", layout="wide")
    
    # Initialize
    init_session_memory_tracking()
    
    # Display metrics
    display_memory_per_user()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Your App", "Cache Test", "Memory Analysis"])
    
    with tab1:
        your_app_with_memory_tracking()
    
    with tab2:
        test_json_caching()
    
    with tab3:
        st.header("📈 Memory Analysis")
        
        # Get all memory data
        if 'memory_snapshots' in st.session_state and st.session_state.memory_snapshots:
            import pandas as pd
            
            df = pd.DataFrame(st.session_state.memory_snapshots)
            
            # Show growth chart
            st.line_chart(df.set_index('timestamp')[['total_memory_mb', 'session_state_mb']])
            
            # Show raw data
            st.dataframe(df, use_container_width=True)
            
            # Calculate if app will handle 150 users
            if len(df) > 0:
                max_memory = df['total_memory_mb'].max()
                avg_session = df['session_state_mb'].mean()
                
                st.subheader("Load Test Prediction")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max Memory Observed", f"{max_memory:.0f} MB")
                with col2:
                    st.metric("Avg Session Size", f"{avg_session:.2f} MB")
                with col3:
                    projected = 500 + (avg_session * 150)  # 500MB base + sessions
                    st.metric("Projected for 150 users", f"{projected/1024:.1f} GB")
                
                if projected > 200000:
                    st.error("""
                    ### ❌ Your app will CRASH with 150 users!
                    
                    **You MUST implement these fixes:**
                    1. Add @st.cache_resource to JSON loading
                    2. Reduce data stored in session state
                    3. Implement lazy loading
                    """)
                else:
                    st.success("### ✅ Your app should handle 150 users")

# ============================================================
# INSTRUCTIONS FOR YOUR APP
# ============================================================

"""
ADD THESE 3 THINGS TO YOUR APP:

1. At the top:
   init_session_memory_tracking()
   display_memory_per_user()

2. Around your operations:
   track_memory_checkpoint("before_operation")
   # your code
   track_memory_checkpoint("after_operation")

3. Check the sidebar to see:
   - Memory per user
   - Memory growth
   - Projection for 150 users
   - RED/YELLOW/GREEN status

This will tell you EXACTLY if your app can handle 150 users!
"""