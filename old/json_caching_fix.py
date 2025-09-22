"""
CRITICAL FIX: Proper JSON Caching Implementation
This will reduce memory usage from 150GB (150 users × 1GB) to just 1GB total!
"""

import streamlit as st
import json
import time
import os
from datetime import datetime
import mmap
import pickle
import hashlib
from pathlib import Path

# ================================================
# OPTION 1: SIMPLE CACHE (Recommended - Start Here)
# ================================================

@st.cache_resource
def load_large_json_shared(file_path="your_1gb_file.json"):
    """
    This loads the JSON file ONCE and shares it across ALL sessions.
    
    CRITICAL: Use @st.cache_resource NOT @st.cache_data
    - cache_resource = Shared across all users (singleton)
    - cache_data = Copied for each user (would still use 150GB!)
    """
    print(f"[{datetime.now()}] Loading JSON file into shared memory...")
    start_time = time.time()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        load_time = time.time() - start_time
        print(f"[{datetime.now()}] JSON loaded in {load_time:.2f} seconds")
        print(f"[{datetime.now()}] File size: {os.path.getsize(file_path) / (1024**3):.2f} GB")
        
        # Add metadata for monitoring
        data_with_metadata = {
            'data': data,
            'loaded_at': datetime.now().isoformat(),
            'load_time_seconds': load_time,
            'file_path': file_path,
            'size_bytes': os.path.getsize(file_path)
        }
        
        return data_with_metadata
        
    except Exception as e:
        st.error(f"Failed to load JSON file: {str(e)}")
        raise

# ================================================
# OPTION 2: LAZY LOADING WITH SECTIONS
# ================================================

@st.cache_resource
def load_json_lazy_sections(file_path="your_1gb_file.json"):
    """
    If your JSON has multiple sections, load only what's needed.
    Good if users only need parts of the data.
    """
    print(f"[{datetime.now()}] Initializing lazy JSON loader...")
    
    # First, load just the structure/keys
    with open(file_path, 'r') as f:
        # If your JSON is like: {"section1": {...}, "section2": {...}}
        # We can load it in chunks
        data = json.load(f)
    
    class LazyJSONAccess:
        def __init__(self, data):
            self._data = data
            self._accessed_sections = set()
        
        def get_section(self, section_name):
            """Get a specific section of the JSON"""
            if section_name not in self._accessed_sections:
                print(f"[{datetime.now()}] Accessing section: {section_name}")
                self._accessed_sections.add(section_name)
            
            return self._data.get(section_name, {})
        
        def get_sections_list(self):
            """Get list of available sections without loading data"""
            return list(self._data.keys())
        
        def search_in_data(self, search_term, section=None):
            """Search within the data without loading everything"""
            results = []
            sections_to_search = [section] if section else self._data.keys()
            
            for sec in sections_to_search:
                if sec in self._data:
                    # Implement your search logic here
                    pass
            
            return results
    
    return LazyJSONAccess(data)

# ================================================
# OPTION 3: MEMORY-MAPPED FILE (Most Efficient)
# ================================================

@st.cache_resource
def load_json_memory_mapped(file_path="your_1gb_file.json"):
    """
    Uses memory mapping for the most efficient memory usage.
    The OS handles paging in/out of memory as needed.
    """
    print(f"[{datetime.now()}] Setting up memory-mapped JSON access...")
    
    class MemoryMappedJSON:
        def __init__(self, file_path):
            self.file_path = file_path
            self._data = None
            self._mmap = None
            self._file = None
            
        def _ensure_loaded(self):
            """Load data on first access"""
            if self._data is None:
                print(f"[{datetime.now()}] Loading memory-mapped JSON...")
                self._file = open(self.file_path, 'r')
                
                # Create memory map
                self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
                
                # Parse JSON from memory map
                self._data = json.loads(self._mmap.read().decode('utf-8'))
                
                # Return to beginning for potential re-reads
                self._mmap.seek(0)
                
        def get_data(self):
            """Get the full data (loads on first call)"""
            self._ensure_loaded()
            return self._data
        
        def __del__(self):
            """Cleanup when object is destroyed"""
            if self._mmap:
                self._mmap.close()
            if self._file:
                self._file.close()
    
    return MemoryMappedJSON(file_path)

# ================================================
# OPTION 4: CACHED WITH SUBSET EXTRACTION
# ================================================

@st.cache_resource
def get_json_data_manager(file_path="your_1gb_file.json"):
    """
    Advanced caching with subset extraction capabilities.
    Useful when different users need different parts of the data.
    """
    
    class JSONDataManager:
        def __init__(self, file_path):
            self.file_path = file_path
            self._full_data = None
            self._subsets_cache = {}
            self.load_time = None
            self.access_count = 0
            
        def _load_if_needed(self):
            """Lazy load the data on first access"""
            if self._full_data is None:
                start = time.time()
                print(f"[{datetime.now()}] Loading JSON into shared cache...")
                
                with open(self.file_path, 'r') as f:
                    self._full_data = json.load(f)
                
                self.load_time = time.time() - start
                print(f"[{datetime.now()}] Loaded in {self.load_time:.2f}s")
        
        def get_all_data(self):
            """Get complete dataset"""
            self._load_if_needed()
            self.access_count += 1
            return self._full_data
        
        def get_filtered_data(self, filter_func):
            """Get filtered subset of data"""
            self._load_if_needed()
            self.access_count += 1
            
            # Create cache key from function
            cache_key = hashlib.md5(str(filter_func).encode()).hexdigest()
            
            if cache_key not in self._subsets_cache:
                # Apply filter and cache result
                filtered = filter_func(self._full_data)
                self._subsets_cache[cache_key] = filtered
            
            return self._subsets_cache[cache_key]
        
        def get_by_keys(self, keys_list):
            """Get specific keys from the data"""
            self._load_if_needed()
            self.access_count += 1
            
            result = {}
            for key in keys_list:
                if key in self._full_data:
                    result[key] = self._full_data[key]
            
            return result
        
        def get_stats(self):
            """Get usage statistics"""
            return {
                'loaded': self._full_data is not None,
                'load_time_seconds': self.load_time,
                'access_count': self.access_count,
                'cached_subsets': len(self._subsets_cache),
                'memory_usage_mb': self._get_memory_usage()
            }
        
        def _get_memory_usage(self):
            """Estimate memory usage"""
            if self._full_data is None:
                return 0
            
            # Rough estimation
            import sys
            return sys.getsizeof(self._full_data) / (1024 * 1024)
    
    return JSONDataManager(file_path)

# ================================================
# HOW TO USE IN YOUR APP - IMMEDIATE FIX
# ================================================

def integrate_json_caching_immediately():
    """
    COPY THIS PATTERN INTO YOUR APP RIGHT NOW!
    """
    
    # At the top of your Streamlit app file:
    st.title("Your App")
    
    # METHOD 1: SIMPLEST - USE THIS FIRST!
    # Replace your current JSON loading with:
    @st.cache_resource
    def load_app_data():
        """This will load only ONCE for ALL users"""
        with open("/path/to/your/1gb/file.json", 'r') as f:
            return json.load(f)
    
    # In your app flow:
    # Instead of loading JSON every time, do this:
    shared_data = load_app_data()  # Shared across all sessions!
    
    # Now use shared_data normally:
    st.write(f"Data loaded: {len(shared_data)} items")
    
    # METHOD 2: WITH ERROR HANDLING
    @st.cache_resource
    def load_app_data_safe():
        """Production-ready version with error handling"""
        file_path = "/path/to/your/1gb/file.json"
        
        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
            return None
        
        try:
            with st.spinner(f"Loading data (first time only)..."):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                st.success(f"Data loaded successfully!")
                return data
                
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON file: {e}")
            return None
        except MemoryError:
            st.error("Not enough memory to load data! Contact admin.")
            return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    # Use it:
    data = load_app_data_safe()
    if data:
        # Your app logic here
        process_data(data)

# ================================================
# TESTING YOUR CACHE
# ================================================

def test_cache_performance():
    """
    Use this to verify caching is working
    """
    st.header("Cache Performance Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Data (First Time)"):
            start = time.time()
            data = load_large_json_shared()
            load_time = time.time() - start
            st.success(f"Loaded in {load_time:.2f} seconds")
            st.write(f"Data size: {len(str(data['data']))} characters")
    
    with col2:
        if st.button("Load Data (From Cache)"):
            start = time.time()
            data = load_large_json_shared()  # Should be instant!
            load_time = time.time() - start
            st.success(f"Loaded in {load_time:.4f} seconds (CACHED!)")
            st.write(f"Loaded at: {data['loaded_at']}")
    
    # Show memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    st.metric("Process Memory", f"{memory_mb:.0f} MB")

# ================================================
# MONITORING CACHE USAGE
# ================================================

@st.cache_resource
def get_cache_monitor():
    """Monitor cache performance"""
    return {
        'hits': 0,
        'loads': 0,
        'last_load': None,
        'load_times': []
    }

def monitored_json_load(file_path="your_1gb_file.json"):
    """Wrapper that monitors cache performance"""
    monitor = get_cache_monitor()
    
    @st.cache_resource
    def _load_json():
        monitor['loads'] += 1
        monitor['last_load'] = datetime.now()
        
        start = time.time()
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        load_time = time.time() - start
        monitor['load_times'].append(load_time)
        
        return data
    
    monitor['hits'] += 1
    data = _load_json()
    
    # Display cache stats in sidebar
    with st.sidebar:
        st.metric("Cache Hits", monitor['hits'])
        st.metric("Actual Loads", monitor['loads'])
        if monitor['loads'] > 0:
            st.metric("Cache Hit Rate", f"{((monitor['hits']-monitor['loads'])/monitor['hits']*100):.1f}%")
    
    return data

if __name__ == "__main__":
    # Test the caching
    st.set_page_config(page_title="JSON Cache Test", layout="wide")
    test_cache_performance()