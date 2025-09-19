# Load Testing Metrics Collection Setup

## 🚀 Quick Start (5 minutes)

### Step 1: Add to Your App

Add this to the **very top** of your Streamlit app file:

```python
# Add these imports at the top
from metrics_collector import (
    init_metrics,
    track_operation,
    track_user_action,
    display_metrics_dashboard
)

# Initialize metrics (right after imports)
init_metrics()

# Add dashboard to sidebar (after init_metrics)
display_metrics_dashboard()
```

### Step 2: Wrap Your Key Functions

Just add `@track_operation("operation_name")` before your existing functions:

```python
@track_operation("load_1gb_json")
@st.cache_resource  # Keep your existing decorators!
def load_large_json():
    # Your existing code - no changes needed
    with open('your_file.json', 'r') as f:
        return json.load(f)

@track_operation("azure_openai_call")
def call_llm(prompt):
    # Your existing code - no changes needed
    return openai_client.complete(prompt)

@track_operation("pdf_processing")
def process_pdf(pdf_file):
    # Your existing code - no changes needed
    return process_with_pypdf2(pdf_file)
```

### Step 3: Deploy and Test

1. Copy `metrics_collector.py` to your app directory
2. Add the imports and decorators
3. Deploy to Domino
4. The metrics dashboard will appear in the sidebar automatically!

## 📊 What You'll See

The sidebar will show:
- **Active Sessions**: Current users
- **Memory Usage**: Current and peak
- **Total Requests**: All operations
- **Error Rate**: Percentage of failures
- **CPU/Memory Graphs**: Real-time usage
- **Operation Performance**: Average and max times for each operation
- **Recent Errors**: Last 3 errors

## 🧪 Running Load Tests

### Option 1: Manual Testing (Fastest)
Have your 5 testers each:
1. Run `python load_test_runner.py`
2. Select scenario (1-4)
3. Script opens multiple browser tabs automatically
4. Complete the workflow in each tab

### Option 2: Coordinated Testing
Share this schedule with your 5 testers:

| Time | Tester 1 | Tester 2 | Tester 3 | Tester 4 | Tester 5 | Total |
|------|----------|----------|----------|----------|----------|-------|
| 0:00 | 1 tab | 1 tab | 1 tab | 1 tab | 1 tab | 5 |
| 0:05 | 5 tabs | 5 tabs | 5 tabs | 5 tabs | 5 tabs | 25 |
| 0:10 | 10 tabs | 10 tabs | 10 tabs | 10 tabs | 10 tabs | 50 |
| 0:15 | 20 tabs | 20 tabs | 20 tabs | 20 tabs | 20 tabs | 100 |
| 0:20 | 30 tabs | 30 tabs | 30 tabs | 30 tabs | 30 tabs | 150 |

## 📈 Interpreting Results

### Good Performance (Green)
- Response times < 5 seconds
- Memory < 100 GB
- Error rate < 5%
- CPU < 80%

### Concerning (Yellow)
- Response times 5-15 seconds
- Memory 100-150 GB
- Error rate 5-15%
- CPU 80-95%

### Critical (Red)
- Response times > 15 seconds
- Memory > 150 GB
- Error rate > 15%
- CPU > 95%

## 🔥 Emergency Fixes If Things Break

### If Memory Explodes:
```python
# Add this IMMEDIATELY
@st.cache_resource  # NOT cache_data!
def load_large_json():
    return json.load(open('file.json'))
```

### If LLM Calls Timeout:
```python
# Add timeout and retry
from concurrent.futures import TimeoutError

@track_operation("llm_call_with_timeout")
def safe_llm_call(prompt, timeout=30):
    try:
        # Add timeout to your Azure OpenAI call
        return openai_client.complete(prompt, timeout=timeout)
    except TimeoutError:
        st.error("LLM call timed out - server overloaded")
        return "Service temporarily unavailable"
```

### If Too Many Concurrent Operations:
```python
from threading import Semaphore

# Limit concurrent operations
llm_limiter = Semaphore(10)  # Max 10 concurrent LLM calls

@track_operation("limited_llm_call")
def rate_limited_llm(prompt):
    with llm_limiter:
        return call_llm(prompt)
```

## 📤 Exporting Results

Click "📥 Export Metrics" button in the sidebar to get a JSON file with:
- Complete performance data
- Error logs
- System metrics over time
- Operation statistics

## 🆘 Troubleshooting

**"Module not found: metrics_collector"**
- Make sure `metrics_collector.py` is in the same directory as your app

**"Metrics not showing"**
- Check that `init_metrics()` is called before any Streamlit components
- Ensure `display_metrics_dashboard()` is called in the main flow

**"High memory usage"**
- Your 1GB JSON MUST use `@st.cache_resource` not `@st.cache_data`
- Check for memory leaks in PDF processing

## Next Steps

1. Add the metrics collector to your app NOW (5 minutes)
2. Deploy to Domino (5 minutes)  
3. Run quick test with 1 user to verify it works (5 minutes)
4. Coordinate load test with 5 users (30-45 minutes)
5. Analyze results and make go/no-go decision

---

**Need help?** The integration is designed to be drop-in. Just copy the decorator pattern from the examples above!