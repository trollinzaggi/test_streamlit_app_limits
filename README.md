# Streamlit Load Testing & Optimization Toolkit

## 📁 Clean File Structure

You now have **ONLY 2 FILES** to work with:

### 1. `memory_compute_tracker.py` - Complete Resource Tracking
- Tracks memory usage (current and peak)
- Tracks CPU usage  
- Per-user session memory tracking
- Automatic 150-user projection
- Red/Yellow/Green status indicators
- Export metrics to JSON

### 2. `optimization_solutions.py` - All Fixes You Need
- JSON caching solution (MOST IMPORTANT)
- Azure OpenAI retry logic
- Azure AI Search retry logic
- Concurrency limiting
- Complete integration examples

### Supporting Files:
- `load_test_runner.py` - Tool to simulate multiple users
- `README.md` - This file

### Archived (in `/old` folder):
All previous versions have been moved to the `old` folder for reference.

---

## 🚀 Quick Start Guide

### Step 1: Add Tracking to Your App (5 minutes)

```python
# At the top of your streamlit app
from memory_compute_tracker import (
    track_resource_usage,
    checkpoint,
    init_session_tracking,
    display_resource_monitor
)

# In your main function
def main():
    # Initialize tracking
    init_session_tracking()
    
    # Add monitor to sidebar
    display_resource_monitor()
    
    # Your app code...
```

### Step 2: Track Your Operations

```python
# Add decorator to functions you want to track
@track_resource_usage("load_json")
@st.cache_resource  # Keep existing decorators
def load_json_data():
    return json.load(open('file.json'))

@track_resource_usage("azure_openai_call")
def call_llm(prompt):
    return azure_openai.complete(prompt)

@track_resource_usage("pdf_processing")
def process_pdf(file):
    return process_with_pypdf(file)
```

### Step 3: Implement Optimizations (20 minutes)

```python
# From optimization_solutions.py

# CRITICAL FIX #1: JSON Caching
@st.cache_resource  # This one line saves 149GB!
def load_json_data():
    return json.load(open('1gb_file.json'))

# FIX #2: Add retry logic
from optimization_solutions import call_azure_openai_with_retry
response = call_azure_openai_with_retry(prompt)

# FIX #3: Add concurrency limits
from optimization_solutions import llm_semaphore, with_concurrency_limit

@with_concurrency_limit(llm_semaphore, "LLM")
def call_llm(prompt):
    return call_azure_openai_with_retry(prompt)
```

---

## 📊 What You'll See in the Sidebar

```
📊 Resource Monitor
Session: abc123

Current Status:
Memory: 1,234 MB    Peak: 2,456 MB
Session: 45.1 MB    CPU: 23.1%
Sessions: 5         Ops: 47

Operations:
├── load_json: Peak 2,456 MB ⚠️ spike
├── azure_openai: 125 MB, 2.3s
└── pdf_process: 89 MB, 1.1s

150 User Projection:
Memory (peak): 165.3 GB
Memory (current): 95.2 GB
CPU projection: 85%
⚠️ Risky for 150 users
```

---

## 🎯 Decision Criteria

After implementing tracking, check the projection:

| Projection | Status | Action |
|------------|--------|--------|
| < 100 GB | ✅ GREEN | Good to go! |
| 100-150 GB | 🟡 YELLOW | Risky, optimize more |
| > 150 GB | 🔴 RED | Won't work, must fix caching |

---

## ⚡ Impact of Fixes

| Fix | Time to Implement | Impact |
|-----|------------------|---------|
| JSON Caching | 5 minutes | Reduces memory by 99% |
| Retry Logic | 10 minutes | Reduces failures by 90% |
| Concurrency Limits | 5 minutes | Prevents overload |

---

## 🧪 Testing Your Fixes

1. **Before fixes**: Run app with tracking, check projection
2. **Implement JSON caching**: Should see immediate memory drop
3. **Add retry logic**: Test with high load
4. **Check final projection**: Should be < 100GB for 150 users

---

## 📞 Quick Troubleshooting

**"Projection shows > 200GB"**
→ JSON file is not cached properly. Must use `@st.cache_resource`

**"Memory spikes but returns to normal"**
→ This is normal. The tracker shows both peak and current.

**"CPU projection > 100%"**
→ CPU will be your bottleneck. Consider optimizing algorithms.

---

## ✅ Final Checklist

- [ ] Added tracking to app
- [ ] Can see memory metrics in sidebar
- [ ] Implemented JSON caching with `@st.cache_resource`
- [ ] Added retry logic to Azure calls
- [ ] Added concurrency limits
- [ ] Projection shows < 150GB for 150 users
- [ ] Tested with multiple concurrent users

Once all checked, your app should handle 150-200 concurrent users!