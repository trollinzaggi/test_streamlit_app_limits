"""
MEMORY AND COMPUTE RESOURCE TRACKER FOR STREAMLIT
Tracks resource usage and saves metrics to CSV files in a specified directory
"""

import streamlit as st
import psutil
import time
import threading
import sys
import csv
import os
from datetime import datetime
from typing import Any, Optional, Dict, List
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS TO YOUR DESIRED DIRECTORY PATH
METRICS_OUTPUT_DIR = "/path/to/your/metrics/directory"  # <-- CHANGE THIS PATH

def ensure_metrics_directory():
    """Ensure the metrics output directory exists"""
    if not os.path.exists(METRICS_OUTPUT_DIR):
        os.makedirs(METRICS_OUTPUT_DIR)

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
# CSV FILE MANAGEMENT
# ============================================================

def get_csv_filepath(filename_base="metrics"):
    """Generate CSV filepath with timestamp in the specified directory"""
    ensure_metrics_directory()
    timestamp = datetime.now().strftime("%Y%m%d")
    return os.path.join(METRICS_OUTPUT_DIR, f"{filename_base}_{timestamp}.csv")

def ensure_csv_headers(filepath, headers):
    """Ensure CSV file exists with headers"""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

def write_metrics_to_csv(metrics_dict, csv_type="operations"):
    """Write metrics to appropriate CSV file"""
    
    if csv_type == "operations":
        filepath = get_csv_filepath("operation_metrics")
        headers = [
            'timestamp', 'session_id', 'operation', 'duration_seconds',
            'start_mb', 'end_mb', 'peak_mb', 'peak_increase_mb', 
            'final_increase_mb', 'had_spike', 'avg_cpu_percent', 
            'max_cpu_percent', 'samples_collected'
        ]
    elif csv_type == "checkpoints":
        filepath = get_csv_filepath("checkpoint_metrics")
        headers = [
            'timestamp', 'session_id', 'checkpoint_name',
            'total_memory_mb', 'session_memory_mb', 'cpu_percent'
        ]
    elif csv_type == "summary":
        filepath = get_csv_filepath("summary_metrics")
        headers = [
            'timestamp', 'active_sessions', 'total_operations',
            'current_memory_mb', 'peak_memory_mb', 'cpu_percent',
            'estimated_per_user_mb', 'projection_150_users_gb'
        ]
    else:
        return
    
    ensure_csv_headers(filepath, headers)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(metrics_dict)

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
        
        # Register session globally
        global_metrics = get_global_metrics()
        with global_metrics['lock']:
            global_metrics['active_sessions'].add(st.session_state.session_id)
            global_metrics['session_metrics'][st.session_state.session_id] = {
                'start_time': datetime.now(),
                'operations': 0,
                'peak_memory_mb': 0
            }
        
        # Log session start
        write_metrics_to_csv({
            'timestamp': datetime.now().isoformat(),
            'session_id': st.session_state.session_id,
            'checkpoint_name': 'session_start',
            'total_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024),
            'session_memory_mb': 0,
            'cpu_percent': psutil.cpu_percent(interval=0.1)
        }, csv_type="checkpoints")

def get_session_memory():
    """Calculate memory used by current session state"""
    session_size = 0
    for key, value in st.session_state.items():
        try:
            session_size += sys.getsizeof(value)
            if hasattr(value, '__dict__'):
                session_size += sys.getsizeof(value.__dict__)
            elif isinstance(value, (list, dict, tuple, set)):
                session_size += sum(sys.getsizeof(item) for item in value[:100])  # Sample first 100 items
        except:
            pass
    return session_size / (1024 * 1024)  # Return in MB

# ============================================================
# TRACKING DECORATOR
# ============================================================

def track_resource_usage(operation_name: str):
    """
    Main decorator to track memory and CPU usage with peak detection
    
    Usage:
        @track_resource_usage("load_json")
        def your_function():
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
                
                # Prepare metrics for CSV
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': st.session_state.get('session_id', 'unknown'),
                    'operation': operation_name,
                    'duration_seconds': round(duration, 2),
                    'start_mb': stats['start_mb'],
                    'end_mb': stats['end_mb'],
                    'peak_mb': stats['peak_mb'],
                    'peak_increase_mb': stats['peak_increase_mb'],
                    'final_increase_mb': stats['final_increase_mb'],
                    'had_spike': stats['had_spike'],
                    'avg_cpu_percent': stats['avg_cpu_percent'],
                    'max_cpu_percent': stats['max_cpu_percent'],
                    'samples_collected': stats['samples_collected']
                }
                
                # Write to CSV
                write_metrics_to_csv(metrics, csv_type="operations")
                
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
                    if st.session_state.get('session_id') in global_metrics['session_metrics']:
                        session = global_metrics['session_metrics'][st.session_state.session_id]
                        session['operations'] += 1
                        session['peak_memory_mb'] = max(session['peak_memory_mb'], stats['peak_mb'])
                
                # Increment session operation count
                if 'operation_count' not in st.session_state:
                    st.session_state.operation_count = 0
                st.session_state.operation_count += 1
                
                return result
                
            except Exception as e:
                tracker.stop()
                
                # Log error to CSV
                error_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': st.session_state.get('session_id', 'unknown'),
                    'operation': operation_name,
                    'duration_seconds': time.time() - start_time,
                    'start_mb': tracker.start_mb,
                    'end_mb': 0,
                    'peak_mb': tracker.peak_mb,
                    'peak_increase_mb': 0,
                    'final_increase_mb': 0,
                    'had_spike': False,
                    'avg_cpu_percent': 0,
                    'max_cpu_percent': 0,
                    'samples_collected': 0
                }
                write_metrics_to_csv(error_metrics, csv_type="operations")
                
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
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.get('session_id', 'unknown'),
        'checkpoint_name': name,
        'total_memory_mb': round(current_mb, 1),
        'session_memory_mb': round(session_mb, 2),
        'cpu_percent': round(cpu_percent, 1)
    }
    
    # Write to CSV
    write_metrics_to_csv(checkpoint_data, csv_type="checkpoints")
    
    return checkpoint_data

# ============================================================
# SUMMARY METRICS WRITER
# ============================================================

def write_summary_metrics():
    """Write current summary metrics to CSV"""
    
    # Current system metrics
    process = psutil.Process()
    current_memory_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Get global metrics
    global_metrics = get_global_metrics()
    peak_memory_mb = max(global_metrics['absolute_peak_mb'], current_memory_mb)
    active_sessions = len(global_metrics['active_sessions'])
    
    # Calculate projections
    baseline_mb = 500  # Streamlit base overhead
    if active_sessions > 0:
        per_user_mb = (current_memory_mb - baseline_mb) / active_sessions
    else:
        per_user_mb = 0
    
    projection_150_users_gb = (baseline_mb + per_user_mb * 150) / 1024
    
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'active_sessions': active_sessions,
        'total_operations': global_metrics['total_operations'],
        'current_memory_mb': round(current_memory_mb, 1),
        'peak_memory_mb': round(peak_memory_mb, 1),
        'cpu_percent': round(cpu_percent, 1),
        'estimated_per_user_mb': round(per_user_mb, 1),
        'projection_150_users_gb': round(projection_150_users_gb, 2)
    }
    
    write_metrics_to_csv(summary_data, csv_type="summary")
    
    return summary_data

# ============================================================
# REPORT GENERATION (CAN BE CALLED FROM EXTERNAL SCRIPT)
# ============================================================

def generate_load_test_report(metrics_dir=None, output_dir=None):
    """
    Generate comprehensive load test report from CSV files
    
    Args:
        metrics_dir: Directory containing the CSV files (defaults to METRICS_OUTPUT_DIR)
        output_dir: Directory to save the report (defaults to metrics_dir)
    
    Returns:
        Path to the generated report file
    
    Can be called from Jupyter notebook:
        from memory_compute_tracker import generate_load_test_report
        report = generate_load_test_report(
            metrics_dir="/path/to/metrics",
            output_dir="/path/to/reports"
        )
    """
    
    # Use provided directory or default
    if metrics_dir is None:
        metrics_dir = METRICS_OUTPUT_DIR
    
    if output_dir is None:
        output_dir = metrics_dir
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(output_dir, f"load_test_report_{report_timestamp}.txt")
    
    # Get date for CSV files
    csv_date = datetime.now().strftime("%Y%m%d")
    
    with open(report_filename, 'w') as report:
        report.write("STREAMLIT LOAD TEST REPORT\n")
        report.write("=" * 50 + "\n")
        report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"Metrics Directory: {metrics_dir}\n\n")
        
        # Read and analyze operation metrics
        operations_file = os.path.join(metrics_dir, f"operation_metrics_{csv_date}.csv")
        if os.path.exists(operations_file):
            df_ops = pd.read_csv(operations_file)
            
            report.write("OPERATION STATISTICS\n")
            report.write("-" * 30 + "\n")
            
            if not df_ops.empty:
                for operation in df_ops['operation'].unique():
                    op_data = df_ops[df_ops['operation'] == operation]
                    report.write(f"\nOperation: {operation}\n")
                    report.write(f"  Total calls: {len(op_data)}\n")
                    report.write(f"  Avg duration: {op_data['duration_seconds'].mean():.2f} seconds\n")
                    report.write(f"  Max duration: {op_data['duration_seconds'].max():.2f} seconds\n")
                    report.write(f"  Avg memory increase: {op_data['final_increase_mb'].mean():.1f} MB\n")
                    report.write(f"  Peak memory: {op_data['peak_mb'].max():.1f} MB\n")
                    report.write(f"  Avg CPU: {op_data['avg_cpu_percent'].mean():.1f}%\n")
                    report.write(f"  Max CPU: {op_data['max_cpu_percent'].max():.1f}%\n")
                    report.write(f"  Memory spikes: {op_data['had_spike'].sum()} occurrences\n")
            else:
                report.write("No operation data found\n")
        else:
            report.write(f"Operation metrics file not found: {operations_file}\n")
        
        # Read and analyze checkpoint metrics
        checkpoint_file = os.path.join(metrics_dir, f"checkpoint_metrics_{csv_date}.csv")
        if os.path.exists(checkpoint_file):
            df_check = pd.read_csv(checkpoint_file)
            
            report.write("\n\nCHECKPOINT STATISTICS\n")
            report.write("-" * 30 + "\n")
            
            if not df_check.empty:
                report.write(f"Total checkpoints: {len(df_check)}\n")
                report.write(f"Unique sessions: {df_check['session_id'].nunique()}\n")
                report.write(f"Max memory at checkpoint: {df_check['total_memory_mb'].max():.1f} MB\n")
                report.write(f"Avg memory at checkpoints: {df_check['total_memory_mb'].mean():.1f} MB\n")
        
        # Read and analyze summary metrics
        summary_file = os.path.join(metrics_dir, f"summary_metrics_{csv_date}.csv")
        if os.path.exists(summary_file):
            df_summary = pd.read_csv(summary_file)
            
            report.write("\n\nSUMMARY STATISTICS\n")
            report.write("-" * 30 + "\n")
            
            if not df_summary.empty:
                report.write(f"Total summary records: {len(df_summary)}\n")
                report.write(f"Max concurrent sessions: {df_summary['active_sessions'].max()}\n")
                report.write(f"Total operations: {df_summary['total_operations'].max()}\n")
                report.write(f"Peak memory: {df_summary['peak_memory_mb'].max():.1f} MB\n")
                report.write(f"Avg memory: {df_summary['current_memory_mb'].mean():.1f} MB\n")
                report.write(f"Max CPU: {df_summary['cpu_percent'].max():.1f}%\n")
                report.write(f"Avg CPU: {df_summary['cpu_percent'].mean():.1f}%\n")
                
                report.write("\n\nLOAD CAPACITY PROJECTION\n")
                report.write("-" * 30 + "\n")
                
                # Get latest projection
                latest = df_summary.iloc[-1]
                projection_gb = latest['projection_150_users_gb']
                report.write(f"Based on latest metrics:\n")
                report.write(f"  Estimated per user: {latest['estimated_per_user_mb']:.1f} MB\n")
                report.write(f"  Projection for 150 users: {projection_gb:.2f} GB\n")
                
                # Average projection
                avg_projection = df_summary['projection_150_users_gb'].mean()
                report.write(f"\nAverage projection: {avg_projection:.2f} GB\n")
                
                # Max projection (worst case)
                max_projection = df_summary['projection_150_users_gb'].max()
                report.write(f"Worst-case projection: {max_projection:.2f} GB\n")
                
                report.write("\nASSESSMENT:\n")
                if max_projection > 200:
                    report.write("Status: FAIL - Will not handle 150 users\n")
                    report.write("Recommendation: Implement caching optimizations immediately\n")
                elif max_projection > 150:
                    report.write("Status: WARNING - May struggle with 150 users\n")
                    report.write("Recommendation: Monitor closely and optimize further\n")
                else:
                    report.write("Status: PASS - Should handle 150 users\n")
                    report.write("Recommendation: Proceed with deployment\n")
            else:
                report.write("No summary data found\n")
        else:
            report.write(f"Summary metrics file not found: {summary_file}\n")
    
    print(f"Report generated: {report_filename}")
    return report_filename

# ============================================================
# PERIODIC SUMMARY WRITER
# ============================================================

@st.cache_resource
def get_summary_writer():
    """Background thread that periodically writes summary metrics"""
    class SummaryWriter:
        def __init__(self, interval=30):  # Write every 30 seconds
            self.interval = interval
            self.running = False
            self.thread = None
            
        def start(self):
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._write_loop, daemon=True)
                self.thread.start()
        
        def stop(self):
            self.running = False
            
        def _write_loop(self):
            while self.running:
                try:
                    write_summary_metrics()
                except:
                    pass
                time.sleep(self.interval)
    
    return SummaryWriter()

# ============================================================
# JUPYTER NOTEBOOK USAGE EXAMPLE
# ============================================================

"""
USAGE FROM JUPYTER NOTEBOOK:

# Cell 1: Import and generate report
from memory_compute_tracker import generate_load_test_report

# Generate report from specific directory
report = generate_load_test_report(
    metrics_dir="/path/to/your/metrics/directory",
    output_dir="/path/to/save/reports"
)
print(f"Report saved: {report}")

# Cell 2: Analyze CSV files directly
import pandas as pd
import matplotlib.pyplot as plt

# Read operation metrics
df_ops = pd.read_csv("/path/to/metrics/operation_metrics_20241220.csv")
df_summary = pd.read_csv("/path/to/metrics/summary_metrics_20241220.csv")

# Plot memory over time
plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(df_summary['timestamp']), df_summary['current_memory_mb'])
plt.xlabel('Time')
plt.ylabel('Memory (MB)')
plt.title('Memory Usage Over Time')
plt.show()

# Analyze operations
print(df_ops.groupby('operation').agg({
    'duration_seconds': ['mean', 'max'],
    'peak_mb': 'max',
    'had_spike': 'sum'
}))
"""

if __name__ == "__main__":
    # Example: Generate report for testing
    print(f"Metrics will be saved to: {METRICS_OUTPUT_DIR}")
    print("Remember to update METRICS_OUTPUT_DIR in this file!")
    
    # Test report generation
    # report = generate_load_test_report()
    # print(f"Test report generated: {report}")