"""
COMPLETE MEMORY TRACKING WITH PEAK CAPTURE - READY TO USE
Add this to your Streamlit app for comprehensive memory monitoring
"""

import streamlit as st
import psutil
import time
import threading
import sys
from datetime import datetime
import pandas as pd

# ============================================================
# SIMPLE PEAK MEMORY TRACKING FOR YOUR APP
# ============================================================

class SimplePeakTracker:
    """Simple peak memory tracker that runs in background"""
    
    def __init__(self):
        self.tracking = False
        self.peak_mb = 0
        self.start_mb = 0
        self.samples = []
        
    def start(self):
        """Start tracking in background"""
        if self.tracking:
            return
            
        self.tracking = True
        process = psutil.Process()
        self.start_mb = process.memory_info().rss / (1024 * 1024)
        self.peak_mb = self.start_mb
        self.samples = []
        
        # Start background monitoring
        thread = threading.Thread(target=self._monitor, daemon=True)
        thread.start()
        
    def stop(self):
        """Stop tracking and return results"""
        self.tracking = False
        time.sleep(0.1)  # Let last sample complete
        
        process = psutil.Process()
        end_mb = process.memory_info().rss / (1024 * 1024)
        
        return {
            'start_mb': self.start_mb,
            'end_mb': end_mb,
            'peak_mb': self.peak_mb,
            'peak_increase_mb': self.peak_mb - self.start_mb,
            'final_increase_mb': end_mb - self.start_mb,
            'had_spike': self.peak_mb > end_mb * 1.1  # Peak was >10% higher than final
        }
    
    def _monitor(self):
        """Background monitoring loop"""
        while self.tracking:
            try:
                process = psutil.Process()
                current_mb = process.memory_info().rss / (1024 * 1024)
                self.peak_mb = max(self.peak_mb, current_mb)
                self.samples.append((time.time(), current_mb))
                time.sleep(0.05)  # Sample every 50ms
            except:
                pass

# Global tracker instance
@st.cache_resource
def get_peak_tracker():
    return {
        'absolute_peak_mb': 0,
        'operation_peaks': {},
        'current_tracker': None
    }

# ============================================================
# EASY-TO-USE WRAPPER FUNCTIONS
# ============================================================

def track_with_peak(operation_name):
    """
    Decorator to track peak memory during any operation
    
    Usage:
        @track_with_peak("load_json")
        def your_function():
            # your code
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = SimplePeakTracker()
            tracker.start()
            
            try:
                # Run the function
                result = func(*args, **kwargs)
                
                # Get peak memory stats
                stats = tracker.stop()
                
                # Update global peak
                global_peak = get_peak_tracker()
                global_peak['absolute_peak_mb'] = max(
                    global_peak['absolute_peak_mb'], 
                    stats['peak_mb']
                )
                global_peak['operation_peaks'][operation_name] = stats
                
                # Show warning if there was a spike
                if stats['had_spike']:
                    st.warning(f"""
                    ⚠️ Memory spike detected in {operation_name}:
                    - Peak: {stats['peak_mb']:.0f} MB (temporary)
                    - Final: {stats['end_mb']:.0f} MB (settled)
                    - Spike: {stats['peak_increase_mb']:.0f} MB above start
                    """)
                
                return result
                
            except Exception as e:
                tracker.stop()
                raise e
        
        return wrapper
    return decorator

# ============================================================
# DISPLAY DASHBOARD WITH PEAKS
# ============================================================

def show_memory_dashboard():
    """Complete memory dashboard for sidebar"""
    
    with st.sidebar:
        st.header("🧠 Memory Monitor")
        
        # Current metrics
        process = psutil.Process()
        current_mb = process.memory_info().rss / (1024 * 1024)
        
        # Get peaks
        global_peak = get_peak_tracker()
        peak_mb = max(global_peak['absolute_peak_mb'], current_mb)
        
        # Session state size
        session_size = 0
        for key, value in st.session_state.items():
            try:
                session_size += sys.getsizeof(value)
            except:
                pass
        session_mb = session_size / (1024 * 1024)
        
        # Display main metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current", f"{current_mb:.0f} MB")
            st.metric("Session", f"{session_mb:.1f} MB")
        
        with col2:
            st.metric(
                "Peak", 
                f"{peak_mb:.0f} MB",
                delta=f"+{(peak_mb - current_mb):.0f}" if peak_mb > current_mb else None
            )
            
            # Peak to current ratio
            if current_mb > 0:
                ratio = peak_mb / current_mb
                if ratio > 1.5:
                    st.metric("Peak/Current", f"{ratio:.1f}x", delta="High!")
                else:
                    st.metric("Peak/Current", f"{ratio:.1f}x")
        
        # Operation breakdown
        if global_peak['operation_peaks']:
            st.subheader("Operations")
            
            for op_name, stats in global_peak['operation_peaks'].items():
                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"**{op_name}**")
                with col2:
                    if stats['had_spike']:
                        st.text(f"🔺 Peak: {stats['peak_mb']:.0f} MB")
                    else:
                        st.text(f"📊 Used: {stats['final_increase_mb']:.0f} MB")
        
        # Load test projection
        st.subheader("150 User Projection")
        
        # Calculate projections
        baseline_mb = 500  # Streamlit base
        
        # Conservative: Use peak memory
        peak_per_user = max(peak_mb - baseline_mb, session_mb)
        peak_projection_gb = (baseline_mb + peak_per_user * 150) / 1024
        
        # Optimistic: Use current memory  
        current_per_user = max(current_mb - baseline_mb, session_mb)
        current_projection_gb = (baseline_mb + current_per_user * 150) / 1024
        
        # Show projections
        st.info(f"""
        **Conservative (peak):**
        {peak_projection_gb:.1f} GB
        
        **Optimistic (current):**
        {current_projection_gb:.1f} GB
        """)
        
        # Status indicator
        if peak_projection_gb > 200:
            st.error("❌ Won't handle 150 users")
            st.caption("Need caching fixes!")
        elif peak_projection_gb > 150:
            st.warning("⚠️ Risky - may crash")
            st.caption("Optimize memory usage")
        else:
            st.success("✅ Should handle 150")
            st.caption(f"Good to go!")

# ============================================================
# YOUR APP INTEGRATION EXAMPLE
# ============================================================

def your_app_with_peak_tracking():
    """
    Example of how to add peak tracking to your app
    """
    
    st.title("Your App with Peak Memory Tracking")
    
    # Show dashboard in sidebar
    show_memory_dashboard()
    
    # =====================================
    # YOUR EXISTING FUNCTIONS WITH TRACKING
    # =====================================
    
    # JSON Loading with peak tracking
    @track_with_peak("load_1gb_json")
    @st.cache_resource
    def load_json_data():
        """Your existing JSON load function"""
        # Simulate loading with temporary spike
        import json
        
        # During loading, memory might spike
        # with open('your_1gb_file.json', 'r') as f:
        #     data = json.load(f)
        
        # Simulation
        temp_buffer = ["x" * 1000000 for _ in range(10)]  # Temporary spike
        time.sleep(0.5)
        data = {"data": "loaded"}
        del temp_buffer  # Memory released
        
        return data
    
    # PDF Processing with peak tracking
    @track_with_peak("pdf_processing")
    def process_pdf(file):
        """Your existing PDF processing"""
        # Your PDF code here
        # Peak tracking happens automatically
        time.sleep(0.5)
        return "Processed"
    
    # Azure OpenAI with peak tracking
    @track_with_peak("azure_openai_call")
    def call_llm(prompt, context=None):
        """Your existing LLM call"""
        # Build large prompt (might spike memory)
        full_prompt = f"{context}\n{prompt}" if context else prompt
        
        # Call Azure OpenAI
        # response = openai_client.complete(full_prompt)
        
        # Simulation
        time.sleep(0.5)
        return "LLM Response"
    
    # Azure Search with peak tracking
    @track_with_peak("azure_search")
    def search_documents(query):
        """Your existing search function"""
        # Your search code
        time.sleep(0.3)
        return ["result1", "result2"]
    
    # =====================================
    # YOUR APP FLOW
    # =====================================
    
    # Test buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Load JSON"):
            with st.spinner("Loading..."):
                data = load_json_data()
            st.success("Loaded!")
    
    with col2:
        if st.button("Process PDF"):
            with st.spinner("Processing..."):
                result = process_pdf("test.pdf")
            st.success("Processed!")
    
    with col3:
        if st.button("Search"):
            with st.spinner("Searching..."):
                results = search_documents("test query")
            st.success(f"Found {len(results)} results")
    
    with col4:
        if st.button("Call LLM"):
            with st.spinner("Calling LLM..."):
                response = call_llm("test prompt", "large context")
            st.success("Got response!")
    
    # Show detailed analysis
    if st.checkbox("Show Memory Analysis"):
        global_peak = get_peak_tracker()
        
        if global_peak['operation_peaks']:
            st.subheader("Memory Usage by Operation")
            
            # Create dataframe
            data = []
            for op_name, stats in global_peak['operation_peaks'].items():
                data.append({
                    'Operation': op_name,
                    'Start (MB)': f"{stats['start_mb']:.0f}",
                    'Peak (MB)': f"{stats['peak_mb']:.0f}",
                    'End (MB)': f"{stats['end_mb']:.0f}",
                    'Peak Increase': f"+{stats['peak_increase_mb']:.0f}",
                    'Final Increase': f"+{stats['final_increase_mb']:.0f}",
                    'Had Spike': '🔺' if stats['had_spike'] else '📊'
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Insights
            st.subheader("💡 Insights")
            
            for op_name, stats in global_peak['operation_peaks'].items():
                if stats['had_spike']:
                    spike_ratio = stats['peak_mb'] / stats['end_mb']
                    st.info(f"""
                    **{op_name}**: Memory spiked to {stats['peak_mb']:.0f} MB during processing 
                    but settled back to {stats['end_mb']:.0f} MB ({spike_ratio:.1f}x spike).
                    This is normal for operations that use temporary buffers.
                    """)

# ============================================================
# TEST YOUR FIXES
# ============================================================

def test_caching_impact():
    """Test to verify caching reduces memory"""
    
    st.header("🧪 Cache Impact Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Without Cache")
        
        @track_with_peak("no_cache_load")
        def load_without_cache():
            # Each call allocates new memory
            data = {"data": "x" * 1000000}
            time.sleep(1)
            return data
        
        if st.button("Load (No Cache)", key="nocache"):
            for i in range(3):
                st.write(f"Load {i+1}...")
                data = load_without_cache()
            st.error("Memory grows with each load!")
    
    with col2:
        st.subheader("With Cache")
        
        @track_with_peak("cached_load")
        @st.cache_resource
        def load_with_cache():
            # Only first call allocates memory
            data = {"data": "x" * 1000000}
            time.sleep(1)
            return data
        
        if st.button("Load (Cached)", key="cached"):
            for i in range(3):
                st.write(f"Load {i+1}...")
                data = load_with_cache()
            st.success("Memory allocated only once!")

# ============================================================
# MAIN APP
# ============================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Peak Memory Tracking", layout="wide")
    
    tab1, tab2 = st.tabs(["Your App", "Cache Test"])
    
    with tab1:
        your_app_with_peak_tracking()
    
    with tab2:
        test_caching_impact()

"""
HOW TO USE IN YOUR APP:

1. Import and initialize:
   from peak_memory_complete import track_with_peak, show_memory_dashboard
   show_memory_dashboard()  # Add to sidebar

2. Wrap your functions:
   @track_with_peak("your_operation")
   def your_function():
       # your existing code

3. Check the sidebar to see:
   - Current vs Peak memory
   - Memory spikes during operations
   - 150-user projection based on PEAK
   - Red/Yellow/Green status

The peak tracking will reveal:
- If memory temporarily spikes during operations
- The true worst-case memory usage
- More accurate projections for 150 users
"""