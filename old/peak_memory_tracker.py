"""
ENHANCED MEMORY TRACKING WITH PEAK MEMORY CAPTURE
Tracks peak memory usage during operations, not just start/end difference
"""

import streamlit as st
import psutil
import time
import threading
import sys
from datetime import datetime
from collections import deque
import json

# ============================================================
# PEAK MEMORY MONITORING
# ============================================================

class PeakMemoryMonitor:
    """Continuously monitors memory to capture peak usage during operations"""
    
    def __init__(self, sampling_interval=0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.peak_memory = 0
        self.start_memory = 0
        self.end_memory = 0
        self.memory_samples = []
        self._monitor_thread = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start monitoring memory in background thread"""
        with self._lock:
            if self.monitoring:
                return
            
            self.monitoring = True
            self.memory_samples = []
            self.start_memory = psutil.Process().memory_info().rss
            self.peak_memory = self.start_memory
            
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop(self):
        """Stop monitoring and return statistics"""
        with self._lock:
            self.monitoring = False
            
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
        
        self.end_memory = psutil.Process().memory_info().rss
        
        return {
            'start_memory_mb': self.start_memory / (1024 * 1024),
            'end_memory_mb': self.end_memory / (1024 * 1024),
            'peak_memory_mb': self.peak_memory / (1024 * 1024),
            'peak_increase_mb': (self.peak_memory - self.start_memory) / (1024 * 1024),
            'final_increase_mb': (self.end_memory - self.start_memory) / (1024 * 1024),
            'samples_collected': len(self.memory_samples),
            'memory_timeline': self.memory_samples
        }
    
    def _monitor_loop(self):
        """Background thread that continuously samples memory"""
        while self.monitoring:
            try:
                current_memory = psutil.Process().memory_info().rss
                
                with self._lock:
                    self.peak_memory = max(self.peak_memory, current_memory)
                    self.memory_samples.append({
                        'timestamp': time.time(),
                        'memory_bytes': current_memory,
                        'memory_mb': current_memory / (1024 * 1024)
                    })
                
                time.sleep(self.sampling_interval)
            except:
                pass

@st.cache_resource
def get_global_peak_tracker():
    """Global tracker for peak memory across all operations"""
    return {
        'absolute_peak_mb': 0,
        'operation_peaks': {},
        'session_peaks': {},
        'peak_timestamp': None,
        'lock': threading.Lock()
    }

# ============================================================
# ENHANCED MEMORY TRACKING WITH PEAKS
# ============================================================

def init_enhanced_memory_tracking():
    """Initialize enhanced memory tracking with peak monitoring"""
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.session_start = datetime.now()
        st.session_state.memory_checkpoints = []
        st.session_state.operation_peaks = {}
        st.session_state.current_monitor = None

def track_operation_with_peak(operation_name: str):
    """
    Decorator that tracks peak memory during an operation
    
    Usage:
        @track_operation_with_peak("load_json")
        def load_data():
            return json.load(...)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = PeakMemoryMonitor(sampling_interval=0.05)  # Sample every 50ms
            
            # Store monitor in session state for visibility
            st.session_state.current_monitor = monitor
            
            # Start monitoring
            monitor.start()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Stop monitoring and get stats
                stats = monitor.stop()
                
                # Update global peak tracker
                global_tracker = get_global_peak_tracker()
                with global_tracker['lock']:
                    global_tracker['absolute_peak_mb'] = max(
                        global_tracker['absolute_peak_mb'], 
                        stats['peak_memory_mb']
                    )
                    global_tracker['operation_peaks'][operation_name] = stats['peak_memory_mb']
                    
                    session_id = st.session_state.session_id
                    if session_id not in global_tracker['session_peaks']:
                        global_tracker['session_peaks'][session_id] = {}
                    global_tracker['session_peaks'][session_id][operation_name] = stats
                
                # Store in session state
                if 'operation_peaks' not in st.session_state:
                    st.session_state.operation_peaks = {}
                st.session_state.operation_peaks[operation_name] = stats
                
                # Display peak warning if significant
                if stats['peak_increase_mb'] > 100:  # More than 100MB peak
                    st.warning(f"""
                    ⚠️ High memory peak in {operation_name}:
                    - Peak: {stats['peak_memory_mb']:.0f} MB
                    - Peak increase: {stats['peak_increase_mb']:.0f} MB
                    - Final increase: {stats['final_increase_mb']:.0f} MB
                    """)
                
                return result
                
            except Exception as e:
                monitor.stop()
                raise e
            finally:
                st.session_state.current_monitor = None
        
        return wrapper
    return decorator

def track_memory_with_peak(checkpoint_name: str):
    """
    Enhanced memory checkpoint that also captures current peak
    """
    process = psutil.Process()
    current_memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # Get global peak
    global_tracker = get_global_peak_tracker()
    global_peak = global_tracker['absolute_peak_mb']
    
    # Calculate session state size
    session_size = 0
    for key, value in st.session_state.items():
        try:
            session_size += sys.getsizeof(value)
            if hasattr(value, '__dict__'):
                session_size += sys.getsizeof(value.__dict__)
        except:
            pass
    session_size_mb = session_size / (1024 * 1024)
    
    checkpoint_data = {
        'checkpoint': checkpoint_name,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'current_memory_mb': current_memory_mb,
        'global_peak_mb': max(global_peak, current_memory_mb),
        'session_state_mb': session_size_mb,
        'session_id': st.session_state.session_id
    }
    
    # Update global peak
    global_tracker['absolute_peak_mb'] = max(global_tracker['absolute_peak_mb'], current_memory_mb)
    
    if 'memory_checkpoints' not in st.session_state:
        st.session_state.memory_checkpoints = []
    st.session_state.memory_checkpoints.append(checkpoint_data)
    
    return checkpoint_data

# ============================================================
# DISPLAY WITH PEAK MEMORY
# ============================================================

def display_memory_with_peaks():
    """Enhanced display showing peak memory usage"""
    
    with st.sidebar:
        st.header(f"🧠 Memory Analysis")
        
        # Current and peak memory
        process = psutil.Process()
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        global_tracker = get_global_peak_tracker()
        peak_memory_mb = max(global_tracker['absolute_peak_mb'], current_memory_mb)
        
        # Session info
        session_size = sum(sys.getsizeof(v) for v in st.session_state.values())
        session_mb = session_size / (1024 * 1024)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Current Memory", 
                f"{current_memory_mb:.0f} MB"
            )
            st.metric(
                "Session State", 
                f"{session_mb:.2f} MB"
            )
        
        with col2:
            st.metric(
                "Peak Memory", 
                f"{peak_memory_mb:.0f} MB",
                delta=f"+{(peak_memory_mb - current_memory_mb):.0f} MB from current"
            )
            peak_vs_current_ratio = (peak_memory_mb / current_memory_mb) if current_memory_mb > 0 else 1
            st.metric(
                "Peak/Current Ratio", 
                f"{peak_vs_current_ratio:.2f}x"
            )
        
        # Operation peaks
        if st.session_state.get('operation_peaks'):
            st.subheader("Peak Memory by Operation")
            
            for op_name, stats in st.session_state.operation_peaks.items():
                with st.expander(f"{op_name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Peak", f"{stats['peak_memory_mb']:.0f} MB")
                        st.metric("Start", f"{stats['start_memory_mb']:.0f} MB")
                    with col2:
                        st.metric("Peak Increase", f"+{stats['peak_increase_mb']:.0f} MB")
                        st.metric("Final Increase", f"+{stats['final_increase_mb']:.0f} MB")
                    
                    # Show if peak was temporary
                    if stats['peak_increase_mb'] > stats['final_increase_mb'] * 1.5:
                        st.info(f"💡 Temporary spike: Peak was {stats['peak_increase_mb']:.0f} MB but settled to {stats['final_increase_mb']:.0f} MB")
        
        # Load test projection based on PEAK memory
        st.subheader("📊 Load Test Projection")
        
        # Calculate projection based on peak
        estimated_baseline = 500  # MB for base Streamlit
        estimated_peak_per_user = max(peak_memory_mb - estimated_baseline, session_mb)
        projected_150_users_peak = estimated_baseline + (estimated_peak_per_user * 150)
        
        # Also calculate based on current (for comparison)
        estimated_current_per_user = max(current_memory_mb - estimated_baseline, session_mb)
        projected_150_users_current = estimated_baseline + (estimated_current_per_user * 150)
        
        st.info(f"""
        **Based on PEAK memory:**
        - Peak per user: {estimated_peak_per_user:.0f} MB
        - 150 users (peak): {projected_150_users_peak/1024:.1f} GB
        
        **Based on current memory:**
        - Current per user: {estimated_current_per_user:.0f} MB  
        - 150 users (current): {projected_150_users_current/1024:.1f} GB
        """)
        
        # Status based on PEAK projection (more conservative)
        if projected_150_users_peak > 200000:  # 200 GB
            st.error("❌ Will NOT handle 150 users (based on peak)!")
        elif projected_150_users_peak > 150000:  # 150 GB  
            st.warning("⚠️ Risky for 150 users (based on peak)")
        else:
            st.success("✅ Should handle 150 users")

# ============================================================
# REAL-TIME MEMORY MONITOR
# ============================================================

def show_realtime_memory_monitor():
    """Display real-time memory usage during operations"""
    
    st.subheader("📈 Real-Time Memory Monitor")
    
    # Create placeholder for chart
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # If there's an active monitor, show its data
    if st.session_state.get('current_monitor') and st.session_state.current_monitor.monitoring:
        monitor = st.session_state.current_monitor
        
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Monitoring", "🔴 Active")
            with col2:
                current = psutil.Process().memory_info().rss / (1024 * 1024)
                st.metric("Current", f"{current:.0f} MB")
            with col3:
                peak = monitor.peak_memory / (1024 * 1024)
                st.metric("Peak", f"{peak:.0f} MB")
        
        # Show live chart
        if monitor.memory_samples:
            import pandas as pd
            df = pd.DataFrame(monitor.memory_samples)
            df['time_offset'] = df['timestamp'] - df['timestamp'].iloc[0]
            chart_placeholder.line_chart(df.set_index('time_offset')['memory_mb'])
    else:
        st.info("No active operation. Memory monitor will activate during operations.")

# ============================================================
# EXAMPLE APP WITH PEAK TRACKING
# ============================================================

def example_app_with_peak_tracking():
    """Example showing how to use peak memory tracking"""
    
    st.title("App with Peak Memory Tracking")
    
    # Initialize
    init_enhanced_memory_tracking()
    
    # Display memory metrics
    display_memory_with_peaks()
    
    # Test operations
    st.header("Test Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        @track_operation_with_peak("json_load")
        @st.cache_resource
        def load_json_with_peak():
            # Simulate memory spike during loading
            temp_data = ["x" * 1000000 for _ in range(100)]  # Temporary spike
            time.sleep(1)
            
            # Final data (smaller)
            final_data = {"data": "x" * 1000000}
            del temp_data  # Release temporary memory
            
            return final_data
        
        if st.button("Load JSON"):
            track_memory_with_peak("before_json")
            data = load_json_with_peak()
            track_memory_with_peak("after_json")
            st.success("JSON loaded - check sidebar for peak memory!")
    
    with col2:
        @track_operation_with_peak("pdf_processing")
        def process_pdf_with_peak():
            # Simulate memory spike during processing
            temp_buffer = "x" * 10000000  # 10MB temporary
            time.sleep(0.5)
            processed = temp_buffer[:1000]  # Keep only small result
            del temp_buffer
            return processed
        
        if st.button("Process PDF"):
            track_memory_with_peak("before_pdf")
            result = process_pdf_with_peak()
            track_memory_with_peak("after_pdf")
            st.success("PDF processed - check peak vs final memory!")
    
    with col3:
        @track_operation_with_peak("llm_call")
        def call_llm_with_peak(prompt):
            # Simulate building large context
            context = "context " * 100000  # Large context
            time.sleep(0.5)
            # Return small response
            return "Response"
        
        if st.button("Call LLM"):
            track_memory_with_peak("before_llm")
            response = call_llm_with_peak("test prompt")
            track_memory_with_peak("after_llm")
            st.success("LLM called - check memory spike!")
    
    # Show detailed analysis
    if st.checkbox("Show Detailed Memory Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Memory Checkpoints")
            if st.session_state.get('memory_checkpoints'):
                import pandas as pd
                df = pd.DataFrame(st.session_state.memory_checkpoints)
                st.dataframe(df, use_container_width=True)
        
        with col2:
            st.subheader("Operation Peaks")
            if st.session_state.get('operation_peaks'):
                for op, stats in st.session_state.operation_peaks.items():
                    st.write(f"**{op}:**")
                    st.write(f"- Peak: {stats['peak_memory_mb']:.0f} MB")
                    st.write(f"- Peak increase: {stats['peak_increase_mb']:.0f} MB")
                    st.write(f"- Final increase: {stats['final_increase_mb']:.0f} MB")
                    st.write(f"- Samples: {stats['samples_collected']}")
    
    # Real-time monitor
    show_realtime_memory_monitor()

# ============================================================
# INTEGRATION GUIDE
# ============================================================

def your_app_with_peak_tracking():
    """
    How to integrate peak memory tracking in your app
    """
    
    # 1. Initialize at start
    init_enhanced_memory_tracking()
    display_memory_with_peaks()
    
    # 2. Wrap your functions with peak tracking
    @track_operation_with_peak("load_1gb_json")
    @st.cache_resource
    def load_your_json():
        with open('your_file.json', 'r') as f:
            return json.load(f)
    
    @track_operation_with_peak("azure_openai_call")
    def call_azure_openai(prompt):
        # Your Azure OpenAI code
        # Peak memory will be tracked automatically
        pass
    
    @track_operation_with_peak("pdf_processing")
    def process_pdf(file):
        # Your PDF processing
        # Will capture memory spikes during processing
        pass
    
    # 3. Use checkpoint tracking for manual points
    track_memory_with_peak("app_start")
    
    # Your app logic here
    data = load_your_json()  # Peak tracked automatically
    
    track_memory_with_peak("after_all_operations")

if __name__ == "__main__":
    st.set_page_config(page_title="Peak Memory Tracking", layout="wide")
    example_app_with_peak_tracking()

"""
KEY IMPROVEMENTS:

1. **Peak Memory Capture**: Continuously monitors memory during operations
2. **Peak vs Final**: Shows temporary spikes vs permanent memory increase  
3. **Real-time Monitoring**: Live chart during operations
4. **Conservative Projections**: Uses peak memory for 150-user projections
5. **Operation-level Peaks**: Track peak for each operation separately

This will show you:
- If memory spikes temporarily during operations
- The true maximum memory your app might use
- More accurate projections for 150 users based on peaks
"""