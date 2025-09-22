"""
PER-USER MEMORY TRACKING FOR STREAMLIT
This properly tracks memory usage for each user session
"""

import streamlit as st
import psutil
import time
import gc
import sys
import json
from datetime import datetime
from collections import defaultdict
import threading
import os
import tracemalloc
import pickle
from typing import Any, Dict, List

# ================================================
# CORE MEMORY TRACKING
# ================================================

@st.cache_resource
def get_memory_tracker():
    """Global memory tracker shared across all sessions"""
    return {
        'sessions': {},  # session_id -> memory info
        'baseline_memory': None,
        'start_time': datetime.now(),
        'peak_sessions': 0,
        'peak_total_memory': 0,
        'lock': threading.Lock()
    }

def get_session_id():
    """Get unique session ID for current user"""
    # Streamlit doesn't expose session ID directly, so we create one
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.session_start = datetime.now()
    return st.session_state.session_id

def calculate_session_memory():
    """
    Calculate memory used by current session
    This is an ESTIMATE since Python doesn't track memory per session perfectly
    """
    session_memory = 0
    
    # Method 1: Calculate size of session state
    if hasattr(st, 'session_state'):
        for key, value in st.session_state.items():
            try:
                # Get size of each object in session state
                size = get_deep_size(value)
                session_memory += size
            except:
                # Fallback to simple sizeof
                session_memory += sys.getsizeof(value)
    
    return session_memory

def get_deep_size(obj, seen=None):
    """Calculate the deep size of an object including all referenced objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    # Important: mark as seen *before* entering recursion
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([get_deep_size(v, seen) for v in obj.values()])
        size += sum([get_deep_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum([get_deep_size(i, seen) for i in obj])
        except:
            pass
    
    return size

def track_session_memory():
    """Track memory for current session"""
    tracker = get_memory_tracker()
    session_id = get_session_id()
    
    # Get current process memory
    process = psutil.Process()
    process_memory = process.memory_info()
    
    # Calculate session-specific memory
    session_memory = calculate_session_memory()
    
    # Get number of active sessions
    active_sessions = len(tracker['sessions'])
    
    # Estimate per-session overhead
    if tracker['baseline_memory'] is None:
        tracker['baseline_memory'] = process_memory.rss
    
    # Calculate approximate memory per session
    if active_sessions > 0:
        overhead_per_session = (process_memory.rss - tracker['baseline_memory']) / max(active_sessions, 1)
    else:
        overhead_per_session = 0
    
    with tracker['lock']:
        # Update session info
        tracker['sessions'][session_id] = {
            'session_id': session_id,
            'last_seen': datetime.now(),
            'session_state_size': session_memory,
            'estimated_total_memory': session_memory + overhead_per_session,
            'process_memory_snapshot': process_memory.rss,
            'active_sessions_count': active_sessions,
            'session_duration': (datetime.now() - st.session_state.session_start).total_seconds()
        }
        
        # Update peaks
        tracker['peak_sessions'] = max(tracker['peak_sessions'], active_sessions)
        tracker['peak_total_memory'] = max(tracker['peak_total_memory'], process_memory.rss)
        
        # Clean up old sessions (not seen in last 5 minutes)
        cutoff = datetime.now()
        for sid in list(tracker['sessions'].keys()):
            if (cutoff - tracker['sessions'][sid]['last_seen']).total_seconds() > 300:
                del tracker['sessions'][sid]
    
    return tracker['sessions'][session_id]

# ================================================
# DETAILED MEMORY PROFILING
# ================================================

class MemoryProfiler:
    """Detailed memory profiler for specific operations"""
    
    def __init__(self):
        self.measurements = []
        self.is_tracing = False
        
    def start_profiling(self, operation_name: str):
        """Start memory profiling for an operation"""
        gc.collect()  # Force garbage collection for accurate measurement
        
        self.current_operation = {
            'name': operation_name,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss,
            'start_session_memory': calculate_session_memory()
        }
        
        # Start tracemalloc for detailed tracking
        if not self.is_tracing:
            tracemalloc.start()
            self.is_tracing = True
    
    def end_profiling(self):
        """End memory profiling and record results"""
        if not hasattr(self, 'current_operation'):
            return None
        
        gc.collect()  # Force garbage collection
        
        end_memory = psutil.Process().memory_info().rss
        end_session_memory = calculate_session_memory()
        
        # Get tracemalloc statistics
        if self.is_tracing:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]  # Top 10 memory users
        else:
            top_stats = []
        
        result = {
            'operation': self.current_operation['name'],
            'duration': time.time() - self.current_operation['start_time'],
            'memory_delta_bytes': end_memory - self.current_operation['start_memory'],
            'memory_delta_mb': (end_memory - self.current_operation['start_memory']) / (1024 * 1024),
            'session_memory_delta': end_session_memory - self.current_operation['start_session_memory'],
            'final_memory_mb': end_memory / (1024 * 1024),
            'timestamp': datetime.now().isoformat(),
            'top_memory_lines': [str(stat)[:100] for stat in top_stats[:5]]  # Top 5 for brevity
        }
        
        self.measurements.append(result)
        del self.current_operation
        
        return result

@st.cache_resource
def get_memory_profiler():
    """Get singleton memory profiler"""
    return MemoryProfiler()

# ================================================
# MEMORY TRACKING DECORATOR
# ================================================

def track_memory(operation_name: str):
    """
    Decorator to track memory usage of a function
    
    Usage:
        @track_memory("load_json")
        def load_data():
            return json.load(...)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Track session memory before and after
            session_id = get_session_id()
            profiler = get_memory_profiler()
            
            # Start profiling
            profiler.start_profiling(f"{operation_name}_{session_id}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # End profiling
                profile_result = profiler.end_profiling()
                
                # Store in session state for display
                if 'memory_profiles' not in st.session_state:
                    st.session_state.memory_profiles = []
                st.session_state.memory_profiles.append(profile_result)
                
                # Log if memory increase is significant
                if profile_result and profile_result['memory_delta_mb'] > 10:
                    st.warning(f"⚠️ {operation_name} used {profile_result['memory_delta_mb']:.1f} MB of memory")
                
                return result
                
            except Exception as e:
                profiler.end_profiling()
                raise e
        
        return wrapper
    return decorator

# ================================================
# DISPLAY COMPONENTS
# ================================================

def display_per_user_memory():
    """Display per-user memory metrics in the sidebar"""
    tracker = get_memory_tracker()
    session_id = get_session_id()
    
    # Update current session's memory
    current_session = track_session_memory()
    
    with st.sidebar:
        st.header("🧠 Memory Usage Per User")
        
        # Current session info
        st.subheader(f"Your Session ({session_id})")
        col1, col2 = st.columns(2)
        
        with col1:
            session_state_mb = current_session['session_state_size'] / (1024 * 1024)
            st.metric(
                "Session State", 
                f"{session_state_mb:.2f} MB",
                help="Memory used by your session data"
            )
        
        with col2:
            estimated_total_mb = current_session['estimated_total_memory'] / (1024 * 1024)
            st.metric(
                "Est. Total", 
                f"{estimated_total_mb:.2f} MB",
                help="Estimated total memory for your session"
            )
        
        # Overall statistics
        st.subheader("All Sessions")
        
        total_sessions = len(tracker['sessions'])
        total_process_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Sessions", total_sessions)
            st.metric("Peak Sessions", tracker['peak_sessions'])
        
        with col2:
            st.metric("Total Memory", f"{total_process_mb:.0f} MB")
            if total_sessions > 0:
                avg_per_user = total_process_mb / total_sessions
                st.metric("Avg per User", f"{avg_per_user:.1f} MB")
        
        # Detailed breakdown
        if st.checkbox("Show All Sessions"):
            sessions_data = []
            for sid, info in tracker['sessions'].items():
                sessions_data.append({
                    'Session': sid[:8],
                    'State (MB)': f"{info['session_state_size'] / (1024*1024):.2f}",
                    'Total (MB)': f"{info['estimated_total_memory'] / (1024*1024):.2f}",
                    'Duration (s)': f"{info['session_duration']:.0f}"
                })
            
            if sessions_data:
                st.dataframe(sessions_data, hide_index=True, use_container_width=True)
        
        # Memory profile history
        if 'memory_profiles' in st.session_state and st.session_state.memory_profiles:
            if st.checkbox("Show Memory Operations"):
                recent_ops = st.session_state.memory_profiles[-5:]  # Last 5 operations
                for op in recent_ops:
                    st.text(f"{op['operation']}: {op['memory_delta_mb']:.2f} MB")

def display_memory_analysis():
    """Display detailed memory analysis"""
    tracker = get_memory_tracker()
    
    st.header("📊 Memory Analysis Dashboard")
    
    # Calculate statistics
    if tracker['sessions']:
        session_memories = [s['session_state_size'] / (1024*1024) for s in tracker['sessions'].values()]
        total_memories = [s['estimated_total_memory'] / (1024*1024) for s in tracker['sessions'].values()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min Memory/User", f"{min(total_memories):.1f} MB")
        with col2:
            st.metric("Avg Memory/User", f"{sum(total_memories)/len(total_memories):.1f} MB")
        with col3:
            st.metric("Max Memory/User", f"{max(total_memories):.1f} MB")
        with col4:
            # Project for 150 users
            avg_per_user = sum(total_memories) / len(total_memories)
            projected_150 = avg_per_user * 150
            st.metric("Projected for 150 users", f"{projected_150:.0f} MB")
        
        # Warning if projection exceeds limits
        if projected_150 > 200000:  # 200 GB
            st.error(f"⚠️ Projected memory ({projected_150:.0f} MB) exceeds 200GB limit!")
            st.info("You need to implement caching fixes before handling 150 users.")
        elif projected_150 > 150000:  # 150 GB
            st.warning(f"⚠️ Projected memory ({projected_150:.0f} MB) is concerning. Consider optimizations.")
        else:
            st.success(f"✅ Projected memory ({projected_150:.0f} MB) is within acceptable limits.")

# ================================================
# INTEGRATION WITH YOUR APP
# ================================================

def example_integration():
    """
    Example showing how to integrate memory tracking in your app
    """
    
    # Track JSON loading
    @track_memory("json_load")
    @st.cache_resource  # Still use caching!
    def load_json_with_tracking():
        # Simulate loading 1GB JSON
        data = {"test": "data" * 1000000}  # Simulate large data
        return data
    
    # Track PDF processing
    @track_memory("pdf_processing")
    def process_pdf_with_tracking(pdf_file):
        # Your PDF processing code
        time.sleep(1)  # Simulate processing
        return "Processed"
    
    # Track LLM calls
    @track_memory("llm_call")
    def call_llm_with_tracking(prompt):
        # Your LLM call
        time.sleep(1)  # Simulate API call
        return f"Response to: {prompt}"
    
    # Use in your app
    st.title("App with Memory Tracking")
    
    if st.button("Load JSON"):
        data = load_json_with_tracking()
        st.success("JSON loaded")
    
    if st.button("Process PDF"):
        result = process_pdf_with_tracking("dummy.pdf")
        st.success(result)
    
    if st.button("Call LLM"):
        response = call_llm_with_tracking("Test prompt")
        st.success(response)

# ================================================
# MAIN INTEGRATION GUIDE
# ================================================

"""
HOW TO ADD PER-USER MEMORY TRACKING TO YOUR APP:

1. At the very top of your app, add:
   ```python
   from per_user_memory_tracker import (
       track_memory, 
       display_per_user_memory,
       display_memory_analysis
   )
   ```

2. Add to your sidebar:
   ```python
   display_per_user_memory()
   ```

3. Wrap your memory-intensive functions:
   ```python
   @track_memory("load_json")
   @st.cache_resource
   def load_your_json():
       # your code
   
   @track_memory("pdf_processing")
   def process_pdf(file):
       # your code
   
   @track_memory("llm_call")
   def call_openai(prompt):
       # your code
   ```

4. Add analysis page (optional):
   ```python
   if st.checkbox("Show Memory Analysis"):
       display_memory_analysis()
   ```

This will show:
- Memory used by each user session
- Average memory per user
- Projection for 150 users
- Memory used by each operation
- Warnings if memory usage is too high
"""

if __name__ == "__main__":
    st.set_page_config(page_title="Memory Tracker Test", layout="wide")
    
    # Display memory tracking
    display_per_user_memory()
    
    # Main area
    tab1, tab2 = st.tabs(["Test Operations", "Memory Analysis"])
    
    with tab1:
        example_integration()
    
    with tab2:
        display_memory_analysis()