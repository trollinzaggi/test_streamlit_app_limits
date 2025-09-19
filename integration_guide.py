"""
Integration Guide for Metrics Collection in Your Streamlit Apps

HOW TO ADD TO YOUR EXISTING STREAMLIT APP:
==========================================

Step 1: Add to your main app file
----------------------------------
"""

import streamlit as st
import json
import time
from metrics_collector import (
    init_metrics,
    track_operation,
    track_user_action,
    display_metrics_dashboard,
    capture_system_metrics,
    export_metrics_to_file
)

# Initialize metrics at the very start of your app
init_metrics()

# Display metrics dashboard in sidebar (always visible)
display_metrics_dashboard()

# ============================================
# WRAP YOUR EXISTING FUNCTIONS LIKE THIS:
# ============================================

# EXAMPLE 1: JSON Loading
@track_operation("load_1gb_json")
@st.cache_resource  # Keep your existing cache!
def load_large_json():
    """Your existing JSON loading function"""
    track_user_action("json_load_started")
    
    # YOUR EXISTING CODE HERE
    with open('your_1gb_file.json', 'r') as f:
        data = json.load(f)
    
    track_user_action("json_load_completed", {'size_mb': 1024})
    return data

# EXAMPLE 2: MSAL Authentication
@track_operation("msal_authentication")
def authenticate_user():
    """Your existing MSAL auth function"""
    track_user_action("auth_started")
    
    # YOUR EXISTING MSAL CODE HERE
    # result = msal_app.acquire_token_silent(...)
    
    track_user_action("auth_completed")
    # return result

# EXAMPLE 3: PDF Processing
@track_operation("pdf_processing", capture_args=True)
def process_pdf(pdf_file):
    """Your existing PDF processing"""
    track_user_action("pdf_processing_started", {'filename': pdf_file.name if hasattr(pdf_file, 'name') else 'unknown'})
    
    # YOUR EXISTING PYPDF2 OR AZURE DOC INTELLIGENCE CODE
    # Example with Azure Doc Intelligence:
    # result = document_intelligence_client.begin_analyze_document(...)
    
    track_user_action("pdf_processing_completed")
    # return result

# EXAMPLE 4: Azure AI Search
@track_operation("azure_ai_search")
def search_documents(query):
    """Your existing Azure AI Search function"""
    track_user_action("search_started", {'query_length': len(query)})
    
    # YOUR EXISTING AZURE SEARCH CODE
    # results = search_client.search(query)
    
    track_user_action("search_completed")
    # return results

# EXAMPLE 5: LLM Call to Azure OpenAI
@track_operation("azure_openai_llm_call")
def call_llm(prompt, context=None):
    """Your existing LLM call function"""
    track_user_action("llm_call_started", {
        'prompt_tokens': len(prompt.split()),
        'has_context': context is not None
    })
    
    # YOUR EXISTING AZURE OPENAI CODE
    # response = openai_client.completions.create(
    #     model="your-model",
    #     prompt=prompt,
    #     max_tokens=1000
    # )
    
    track_user_action("llm_call_completed")
    # return response

# ============================================
# MAIN APP FLOW WITH METRICS
# ============================================

def main():
    st.title("Your App Title")
    
    # Track page loads
    track_user_action("page_load")
    
    # YOUR EXISTING UI CODE
    
    # Example: File uploader with tracking
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        track_user_action("file_uploaded", {'size': uploaded_file.size})
        
        # Process with metrics tracking
        with st.spinner("Processing PDF..."):
            # This function is already wrapped with @track_operation
            result = process_pdf(uploaded_file)
    
    # Example: User input with tracking
    user_query = st.text_input("Enter your query")
    if st.button("Submit"):
        track_user_action("query_submitted", {'query_length': len(user_query)})
        
        with st.spinner("Processing..."):
            # These functions are already wrapped
            search_results = search_documents(user_query)
            llm_response = call_llm(user_query, context=search_results)
            
        st.write(llm_response)
    
    # Add export button at the bottom
    if st.button("📊 Generate Load Test Report"):
        filename = export_metrics_to_file()
        st.success(f"Report saved to {filename}")
        
        # Show summary
        metrics = st.session_state.global_metrics
        st.write(f"""
        ### Load Test Summary
        - Total Sessions: {metrics['total_sessions']}
        - Peak Concurrent Sessions: {metrics['peak_sessions']}
        - Total Requests: {metrics['total_requests']}
        - Error Rate: {(metrics['total_errors'] / max(metrics['total_requests'], 1) * 100):.2f}%
        - Peak Memory: {metrics['peak_memory']:.0f} MB
        """)

if __name__ == "__main__":
    main()

# ============================================
# MINIMAL EXAMPLE - QUICK START
# ============================================
"""
If you want to test quickly, just add these 3 lines to your existing app:

1. At the very top after imports:
   from metrics_collector import init_metrics, track_operation, display_metrics_dashboard
   init_metrics()
   display_metrics_dashboard()

2. Wrap your slowest function:
   @track_operation("your_function_name")
   def your_existing_function():
       # your code here

3. That's it! The metrics will start collecting immediately.
"""