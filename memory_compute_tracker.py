"""
COMPLETE MEMORY & COMPUTE TRACKING FOR STREAMLIT
Single file for all tracking and monitoring needs
"""

import streamlit as st
import psutil
import time
import threading
import sys
import json
from datetime import datetime
from typing import Any, Optional, Dict
import pandas as pd

# ============================================================
# CORE MEMORY TRACKER WITH PEAK DETECTION
# ============================================================

class MemoryTracker:
    """Tracks memory usage including peaks during operations"""
    
    def __init__(self, sampling_interval=0.05):
        self.sampling_interval = sampling_interval
        self.tracking = False
        self.peak_mb = 0
        self.start_mb = 0
        self.cpu_samples = []
        self.memory_samples = []
        
    def start(self):
        """Start tracking memory and CPU in background"""
        if self.tracking:
            return
            
        self.tracking = True
        process = psutil.Process()
        self.start_mb = process.memory_info().rss / (1024 * 1024)
        self.peak_mb = self.start_mb
        self.memory_samples = []
        self.cpu_samples = []
        
        thread = threading.Thread(target=self._monitor, daemon=True)
        thread.start()
        
    def stop(self):
        """Stop tracking and return statistics"""
        self.tracking = False
        time.sleep(0.1)  # Let monitoring thread finish
        
        process = psutil.Process()
        end_mb = process.memory_info().rss / (1024 * 1024)
        
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        max_cpu = max(self.cpu_samples) if self.cpu_samples else 0
        
        return {
            'start_mb': round(self.start_mb, 1),
            'end_mb': round(end_mb, 1),
            'peak_mb': round(self.peak_mb, 1),
            'peak_increase_mb': round(self.peak_mb - self.start_mb, 1),
            'final_increase_mb': round(end_mb - self.start_mb, 1),
            'had_spike': self.peak_mb > end_mb * 1.1,
            'avg_cpu_percent': round(avg_cpu, 1),
            'max_cpu_percent': round(max_cpu, 1),
            'samples_collected': len(self.memory_samples)
        }
    
    def _monitor(self):
        """Background monitoring of memory and CPU"""
        process = psutil.Process()
        while self.tracking:
            try:
                # Memory tracking
                current_mb = process.memory_info().rss / (1024 * 1024)
                self.peak_mb = max(self.peak_mb, current_mb)
                self.memory_samples.append(current_mb)
                
                # CPU tracking
                cpu_percent = process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(self.sampling_interval)
            except:
                pass

# ============================================================
# GLOBAL TRACKING STATE
# ============================================================

@st.cache_resource
def get_global_metrics():
    """Global metrics shared across all sessions"""
    return {
        'absolute_peak_mb': 0,
        'total_cpu_seconds': 0,
        'operation_stats': {},
        'session_metrics': {},
        'active_sessions': set(),
        'total_operations': 0,
        'start_time': datetime.now(),
        'lock': threading.Lock()
    }

# ============================================================
# SESSION TRACKING
# ============================================================

def init_session_tracking():
    """Initialize per-session tracking - CALL THIS AT APP START"""
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.session_start = datetime.now()
        st.session_state.operation_count = 0
        st.session_state.memory_checkpoints = []
        st.session_state.operation_history = []
        
        # Register session globally
        global_metrics = get_global_metrics()
        with global_metrics['lock']:
            global_metrics['active_sessions'].add(st.session_state.session_id)
            global_metrics['session_metrics'][st.session_state.session_id] = {
                'start_time': datetime.now(),
                'operations': 0,
                'peak_memory_mb': 0
            }

def get_session_memory():
    """Calculate memory used by current session state"""
    session_size = 0
    for key, value in st.session_state.items():
        try:
            session_size += sys.getsizeof(value)
            # Try to get size of nested objects
            if hasattr(value, '__dict__'):
                session_size += sys.getsizeof(value.__dict__)
            elif isinstance(value, (list, dict, tuple, set)):
                # Rough estimate for collections
                session_size += sum(sys.getsizeof(item) for item in value)
        except:
            pass
    return session_size / (1024 * 1024)  # Return in MB

# ============================================================
# TRACKING DECORATORS
# ============================================================

def track_resource_usage(operation_name: str):
    """
    Main decorator to track memory and CPU usage with peak detection
    
    Usage:
        @track_resource_usage("load_json")
        def your_function():
            # your code
            return data
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Start tracking
            tracker = MemoryTracker()
            start_time = time.time()
            tracker.start()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Stop tracking
                duration = time.time() - start_time
                stats = tracker.stop()
                stats['duration_seconds'] = round(duration, 2)
                stats['operation'] = operation_name
                stats['timestamp'] = datetime.now().isoformat()
                
                # Update global metrics
                global_metrics = get_global_metrics()
                with global_metrics['lock']:
                    global_metrics['absolute_peak_mb'] = max(
                        global_metrics['absolute_peak_mb'], 
                        stats['peak_mb']
                    )
                    global_metrics['operation_stats'][operation_name] = stats
                    global_metrics['total_operations'] += 1
                    global_metrics['total_cpu_seconds'] += duration * (stats['avg_cpu_percent'] / 100)
                    
                    # Update session metrics
                    if st.session_state.session_id in global_metrics['session_metrics']:
                        session = global_metrics['session_metrics'][st.session_state.session_id]
                        session['operations'] += 1
                        session['peak_memory_mb'] = max(session['peak_memory_mb'], stats['peak_mb'])
                
                # Store in session state
                st.session_state.operation_count += 1
                if 'operation_history' not in st.session_state:
                    st.session_state.operation_history = []
                st.session_state.operation_history.append(stats)
                
                # Warnings for high resource usage
                if stats['peak_increase_mb'] > 100:
                    st.warning(f"⚠️ High memory in {operation_name}: {stats['peak_mb']:.0f} MB peak")
                if stats['max_cpu_percent'] > 80:
                    st.warning(f"⚠️ High CPU in {operation_name}: {stats['max_cpu_percent']:.0f}%")
                
                return result
                
            except Exception as e:
                tracker.stop()
                st.error(f"❌ Error in {operation_name}: {str(e)}")
                raise e
        
        return wrapper
    return decorator

def checkpoint(name: str):
    """
    Record a memory checkpoint at any point in your code
    
    Usage:
        checkpoint("before_processing")
        # your code
        checkpoint("after_processing")
    """
    process = psutil.Process()
    current_mb = process.memory_info().rss / (1024 * 1024)
    session_mb = get_session_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    checkpoint_data = {
        'name': name,
        'time': datetime.now().strftime('%H:%M:%S'),
        'total_memory_mb': round(current_mb, 1),
        'session_memory_mb': round(session_mb, 2),
        'cpu_percent': round(cpu_percent, 1)
    }
    
    if 'memory_checkpoints' not in st.session_state:
        st.session_state.memory_checkpoints = []
    st.session_state.memory_checkpoints.append(checkpoint_data)
    
    return checkpoint_data

# ============================================================
# DASHBOARD DISPLAY
# ============================================================

def display_resource_monitor():
    """
    Complete resource monitoring dashboard for sidebar
    CALL THIS IN YOUR SIDEBAR
    """
    
    # Initialize session if needed
    init_session_tracking()
    
    with st.sidebar:
        st.header(f"📊 Resource Monitor")
        st.caption(f"Session: {st.session_state.session_id}")
        
        # Current system metrics
        process = psutil.Process()
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        session_memory_mb = get_session_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get global metrics
        global_metrics = get_global_metrics()
        peak_memory_mb = max(global_metrics['absolute_peak_mb'], current_memory_mb)
        active_sessions = len(global_metrics['active_sessions'])
        
        # Display main metrics
        st.subheader("Current Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Memory", f"{current_memory_mb:.0f} MB")
            st.metric("Session", f"{session_memory_mb:.1f} MB")
        
        with col2:
            st.metric("Peak", f"{peak_memory_mb:.0f} MB")
            st.metric("CPU", f"{cpu_percent:.1f}%")
        
        with col3:
            st.metric("Sessions", active_sessions)
            st.metric("Ops", st.session_state.get('operation_count', 0))
        
        # Memory analysis
        if peak_memory_mb > current_memory_mb * 1.2:
            peak_ratio = peak_memory_mb / current_memory_mb
            st.warning(f"⚠️ Peak was {peak_ratio:.1f}x current")
        
        # Operation breakdown
        if global_metrics['operation_stats']:
            st.subheader("Operations")
            for op_name, stats in global_metrics['operation_stats'].items():
                with st.expander(f"{op_name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Duration: {stats.get('duration_seconds', 0):.1f}s")
                        st.text(f"Memory: +{stats['final_increase_mb']} MB")
                        st.text(f"Peak: {stats['peak_mb']} MB")
                    with col2:
                        st.text(f"CPU Avg: {stats['avg_cpu_percent']}%")
                        st.text(f"CPU Max: {stats['max_cpu_percent']}%")
                        if stats['had_spike']:
                            st.text("⚠️ Memory spike")
        
        # Checkpoints
        if st.session_state.get('memory_checkpoints'):
            if st.checkbox("Show Checkpoints"):
                df = pd.DataFrame(st.session_state.memory_checkpoints)
                st.dataframe(
                    df[['name', 'total_memory_mb', 'cpu_percent']], 
                    hide_index=True,
                    use_container_width=True
                )
        
        # Load test projection
        st.subheader("150 User Projection")
        
        # Calculate projections
        baseline_mb = 500  # Streamlit base
        
        # Conservative (using peak)
        per_user_peak = max(peak_memory_mb - baseline_mb, session_memory_mb) / max(active_sessions, 1)
        projection_peak_gb = (baseline_mb + per_user_peak * 150) / 1024
        
        # Current-based
        per_user_current = max(current_memory_mb - baseline_mb, session_memory_mb) / max(active_sessions, 1)
        projection_current_gb = (baseline_mb + per_user_current * 150) / 1024
        
        # CPU projection
        cpu_per_user = cpu_percent / max(active_sessions, 1)
        projected_cpu = cpu_per_user * 150
        
        st.info(f"""
        **Memory (peak):** {projection_peak_gb:.1f} GB
        **Memory (current):** {projection_current_gb:.1f} GB
        **CPU projection:** {projected_cpu:.0f}%
        """)
        
        # Status indicator
        if projection_peak_gb > 200:
            st.error("❌ Won't handle 150 users")
            st.caption("Need optimization!")
        elif projection_peak_gb > 150:
            st.warning("⚠️ Risky for 150 users")
        elif projected_cpu > 80:
            st.warning("⚠️ CPU may bottleneck")
        else:
            st.success("✅ Should handle 150 users")
        
        # Export button
        if st.button("📥 Export Metrics"):
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'session_id': st.session_state.session_id,
                'current_memory_mb': current_memory_mb,
                'peak_memory_mb': peak_memory_mb,
                'operations': global_metrics['operation_stats'],
                'checkpoints': st.session_state.get('memory_checkpoints', []),
                'projection_150_users_gb': projection_peak_gb
            }
            
            filename = f"metrics_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            st.success(f"Exported to {filename}")

# ============================================================
# INTEGRATION INSTRUCTIONS
# ============================================================

"""
HOW TO INTEGRATE INTO YOUR STREAMLIT APP:

1. IMPORT AT THE TOP OF YOUR APP:
   ```python
   from memory_compute_tracker import (
       track_resource_usage,
       checkpoint,
       init_session_tracking,
       display_resource_monitor
   )
   ```

2. INITIALIZE IN YOUR MAIN FUNCTION:
   ```python
   def main():
       # Initialize tracking
       init_session_tracking()
       
       # Add monitor to sidebar
       display_resource_monitor()
       
       # Your app code...
   ```

3. ADD DECORATOR TO FUNCTIONS YOU WANT TO TRACK:
   ```python
   @track_resource_usage("load_json")
   @st.cache_resource  # Keep your existing decorators
   def load_json_data():
       with open('file.json', 'r') as f:
           return json.load(f)
   
   @track_resource_usage("azure_openai_call")
   def call_llm(prompt):
       # Your LLM code
       return response
   
   @track_resource_usage("pdf_processing")
   def process_pdf(file):
       # Your PDF processing
       return result
   
   @track_resource_usage("azure_search")
   def search_index(query):
       # Your search code
       return results
   ```

4. ADD CHECKPOINTS FOR DETAILED TRACKING (OPTIONAL):
   ```python
   checkpoint("app_start")
   
   # Heavy operation
   data = load_json_data()
   checkpoint("after_json_load")
   
   # Another operation
   results = process_data(data)
   checkpoint("after_processing")
   ```

5. WHAT YOU'LL SEE IN THE SIDEBAR:
   - Current memory usage
   - Peak memory usage  
   - CPU usage
   - Per-operation breakdown
   - Memory spikes detection
   - 150-user projection
   - Red/Yellow/Green status

6. KEY METRICS TO WATCH:
   - Peak Memory > 200GB for 150 users = RED (won't work)
   - Peak Memory 150-200GB = YELLOW (risky)
   - Peak Memory < 150GB = GREEN (should work)
   - CPU projection > 80% = May have CPU bottleneck

The tracker will automatically:
- Detect memory spikes during operations
- Track CPU usage
- Calculate per-user resource needs
- Project capacity for 150 users
- Warn about resource-intensive operations
"""

# ============================================================
# TEST FUNCTION
# ============================================================

def test_tracking():
    """Test function to verify tracking is working"""
    
    @track_resource_usage("test_operation")
    def sample_operation():
        # Simulate memory allocation
        data = ["x" * 1000000 for _ in range(10)]
        time.sleep(0.5)
        return len(data)
    
    # Run test
    checkpoint("before_test")
    result = sample_operation()
    checkpoint("after_test")
    
    st.success(f"Test complete! Processed {result} items. Check sidebar for metrics.")

if __name__ == "__main__":
    st.set_page_config(page_title="Resource Tracker Test", layout="wide")
    
    # Initialize
    init_session_tracking()
    display_resource_monitor()
    
    # Test UI
    st.title("Resource Tracking Test")
    
    if st.button("Run Test"):
        test_tracking()
    
    if st.checkbox("Show Detailed Metrics"):
        global_metrics = get_global_metrics()
        st.json({
            'peak_memory_mb': global_metrics['absolute_peak_mb'],
            'total_operations': global_metrics['total_operations'],
            'active_sessions': len(global_metrics['active_sessions']),
            'operations': global_metrics['operation_stats']
        })