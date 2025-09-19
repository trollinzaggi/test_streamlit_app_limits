"""
Metrics Collection Module for Streamlit Load Testing
Add this to your Streamlit app to capture performance metrics
"""

import streamlit as st
import psutil
import time
import json
import threading
from datetime import datetime
from collections import deque, defaultdict
import pandas as pd
import os
import traceback
from functools import wraps

# Initialize metrics in session state
def init_metrics():
    """Initialize metrics storage in session state"""
    if 'metrics_initialized' not in st.session_state:
        st.session_state.metrics_initialized = True
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        st.session_state.session_start = datetime.now()
        
        # Global metrics (shared across sessions using cache_resource)
        if 'global_metrics' not in st.session_state:
            st.session_state.global_metrics = get_global_metrics()
        
        # Increment active sessions
        st.session_state.global_metrics['active_sessions'].add(st.session_state.session_id)
        st.session_state.global_metrics['total_sessions'] += 1
        
        # Session-specific metrics
        st.session_state.operation_times = []
        st.session_state.errors = []
        st.session_state.user_actions = []

@st.cache_resource
def get_global_metrics():
    """Get or create global metrics shared across all sessions"""
    return {
        'server_start': datetime.now(),
        'active_sessions': set(),
        'total_sessions': 0,
        'total_requests': 0,
        'total_errors': 0,
        'operation_history': deque(maxlen=1000),
        'system_metrics': deque(maxlen=500),
        'error_log': deque(maxlen=100),
        'response_times_by_operation': defaultdict(list),
        'concurrent_operations': defaultdict(int),
        'peak_memory': 0,
        'peak_sessions': 0
    }

def track_operation(operation_name, capture_args=False):
    """
    Decorator to track operation performance
    Usage: @track_operation("load_json")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize metrics if needed
            init_metrics()
            
            # Record operation start
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Track concurrent operations
            global_metrics = st.session_state.global_metrics
            global_metrics['concurrent_operations'][operation_name] += 1
            
            # Operation metadata
            operation_data = {
                'operation': operation_name,
                'session_id': st.session_state.session_id,
                'start_time': datetime.now().isoformat(),
                'start_memory_mb': start_memory
            }
            
            if capture_args:
                # Capture safe representations of arguments
                try:
                    operation_data['args'] = str(args)[:100]  # Limit size
                    operation_data['kwargs'] = str(kwargs)[:100]
                except:
                    pass
            
            try:
                # Execute the actual function
                result = func(*args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                operation_data.update({
                    'duration': duration,
                    'success': True,
                    'end_memory_mb': end_memory,
                    'memory_delta_mb': end_memory - start_memory,
                    'end_time': datetime.now().isoformat()
                })
                
                # Update global metrics
                global_metrics['total_requests'] += 1
                global_metrics['operation_history'].append(operation_data)
                global_metrics['response_times_by_operation'][operation_name].append(duration)
                
                # Update session metrics
                st.session_state.operation_times.append(operation_data)
                
                # Log if operation was slow
                if duration > 5:  # More than 5 seconds
                    st.warning(f"⚠️ Slow operation: {operation_name} took {duration:.2f} seconds")
                
                return result
                
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                error_msg = str(e)
                stack_trace = traceback.format_exc()
                
                operation_data.update({
                    'duration': duration,
                    'success': False,
                    'error': error_msg,
                    'stack_trace': stack_trace,
                    'end_time': datetime.now().isoformat()
                })
                
                # Update global metrics
                global_metrics['total_errors'] += 1
                global_metrics['error_log'].append(operation_data)
                
                # Update session metrics
                st.session_state.errors.append(operation_data)
                
                # Show error in UI
                st.error(f"❌ Error in {operation_name}: {error_msg}")
                
                raise
                
            finally:
                # Decrement concurrent operations
                global_metrics['concurrent_operations'][operation_name] -= 1
                
        return wrapper
    return decorator

def capture_system_metrics():
    """Capture current system metrics"""
    try:
        process = psutil.Process()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'process_memory_mb': process.memory_info().rss / 1024 / 1024,
            'process_cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'active_sessions': len(st.session_state.global_metrics['active_sessions'])
        }
        
        # Update peak values
        global_metrics = st.session_state.global_metrics
        global_metrics['peak_memory'] = max(global_metrics['peak_memory'], metrics['process_memory_mb'])
        global_metrics['peak_sessions'] = max(global_metrics['peak_sessions'], metrics['active_sessions'])
        
        # Store metrics
        global_metrics['system_metrics'].append(metrics)
        
        return metrics
    except Exception as e:
        return {'error': str(e)}

def track_user_action(action_name, details=None):
    """Track user interactions for understanding usage patterns"""
    init_metrics()
    
    action_data = {
        'action': action_name,
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.session_id,
        'details': details
    }
    
    st.session_state.user_actions.append(action_data)

def display_metrics_dashboard():
    """Display real-time metrics dashboard in Streamlit sidebar"""
    with st.sidebar:
        st.header("📊 Performance Metrics")
        
        # Capture current system metrics
        current_metrics = capture_system_metrics()
        
        # Display key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Active Sessions", 
                len(st.session_state.global_metrics['active_sessions']),
                delta=f"Peak: {st.session_state.global_metrics['peak_sessions']}"
            )
            st.metric(
                "Memory (MB)", 
                f"{current_metrics.get('process_memory_mb', 0):.0f}",
                delta=f"Peak: {st.session_state.global_metrics['peak_memory']:.0f}"
            )
        
        with col2:
            st.metric(
                "Total Requests", 
                st.session_state.global_metrics['total_requests']
            )
            error_rate = 0
            if st.session_state.global_metrics['total_requests'] > 0:
                error_rate = (st.session_state.global_metrics['total_errors'] / 
                            st.session_state.global_metrics['total_requests'] * 100)
            st.metric(
                "Error Rate", 
                f"{error_rate:.1f}%",
                delta=f"{st.session_state.global_metrics['total_errors']} errors"
            )
        
        # System resources
        st.subheader("System Resources")
        st.progress(current_metrics.get('cpu_percent', 0) / 100, text=f"CPU: {current_metrics.get('cpu_percent', 0):.1f}%")
        st.progress(current_metrics.get('memory_percent', 0) / 100, text=f"Memory: {current_metrics.get('memory_percent', 0):.1f}%")
        
        # Operation performance
        if st.session_state.global_metrics['response_times_by_operation']:
            st.subheader("Operation Performance")
            perf_data = []
            for op_name, times in st.session_state.global_metrics['response_times_by_operation'].items():
                if times:
                    perf_data.append({
                        'Operation': op_name,
                        'Avg (s)': f"{sum(times)/len(times):.2f}",
                        'Max (s)': f"{max(times):.2f}",
                        'Count': len(times)
                    })
            if perf_data:
                st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)
        
        # Recent errors
        if st.session_state.global_metrics['error_log']:
            st.subheader("Recent Errors")
            recent_errors = list(st.session_state.global_metrics['error_log'])[-3:]
            for error in recent_errors:
                st.error(f"{error['operation']}: {error.get('error', 'Unknown error')[:50]}")
        
        # Export metrics button
        if st.button("📥 Export Metrics"):
            export_metrics_to_file()
            st.success("Metrics exported to metrics_export.json")

def export_metrics_to_file():
    """Export all metrics to a JSON file for analysis"""
    export_data = {
        'export_time': datetime.now().isoformat(),
        'server_uptime': str(datetime.now() - st.session_state.global_metrics['server_start']),
        'summary': {
            'total_sessions': st.session_state.global_metrics['total_sessions'],
            'peak_sessions': st.session_state.global_metrics['peak_sessions'],
            'total_requests': st.session_state.global_metrics['total_requests'],
            'total_errors': st.session_state.global_metrics['total_errors'],
            'error_rate': (st.session_state.global_metrics['total_errors'] / 
                          max(st.session_state.global_metrics['total_requests'], 1) * 100),
            'peak_memory_mb': st.session_state.global_metrics['peak_memory']
        },
        'operation_performance': {},
        'system_metrics': list(st.session_state.global_metrics['system_metrics'])[-100:],  # Last 100 snapshots
        'error_log': list(st.session_state.global_metrics['error_log']),
        'recent_operations': list(st.session_state.global_metrics['operation_history'])[-100:]
    }
    
    # Calculate operation statistics
    for op_name, times in st.session_state.global_metrics['response_times_by_operation'].items():
        if times:
            export_data['operation_performance'][op_name] = {
                'count': len(times),
                'avg_seconds': sum(times) / len(times),
                'min_seconds': min(times),
                'max_seconds': max(times),
                'median_seconds': sorted(times)[len(times)//2]
            }
    
    # Save to file
    filename = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    return filename

def cleanup_session():
    """Clean up session from active sessions on app rerun/close"""
    if 'session_id' in st.session_state:
        try:
            st.session_state.global_metrics['active_sessions'].discard(st.session_state.session_id)
        except:
            pass

# Auto-cleanup on session end
import atexit
atexit.register(cleanup_session)

# Quick test function
def test_metrics_tracking():
    """Test function to verify metrics are working"""
    
    @track_operation("test_operation")
    def sample_operation(delay=1):
        time.sleep(delay)
        return f"Completed in {delay} seconds"
    
    # Run test
    result = sample_operation(0.5)
    st.success(f"Test successful: {result}")
    
    # Display metrics
    st.json({
        'active_sessions': len(st.session_state.global_metrics['active_sessions']),
        'total_requests': st.session_state.global_metrics['total_requests'],
        'last_operation': st.session_state.operation_times[-1] if st.session_state.operation_times else None
    })
