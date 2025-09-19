"""
Quick Load Testing Script - Use this to simulate multiple users
Run this from multiple terminals or machines to simulate concurrent users
"""

import webbrowser
import time
import threading
import subprocess
import sys
from datetime import datetime

class LoadTestCoordinator:
    def __init__(self, app_url, num_tabs=10):
        self.app_url = app_url
        self.num_tabs = num_tabs
        
    def open_tabs_gradually(self, stagger_seconds=2):
        """Open multiple tabs with staggered timing"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting load test with {self.num_tabs} tabs")
        print(f"URL: {self.app_url}")
        print("-" * 50)
        
        for i in range(self.num_tabs):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Opening tab {i+1}/{self.num_tabs}")
            webbrowser.open_new_tab(self.app_url)
            
            if i < self.num_tabs - 1:  # Don't wait after last tab
                print(f"   Waiting {stagger_seconds} seconds before next tab...")
                time.sleep(stagger_seconds)
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All tabs opened!")
        print("=" * 50)
        print("\nNOW COMPLETE THESE STEPS IN EACH TAB:")
        print("1. Login (if required)")
        print("2. Upload a test PDF")
        print("3. Submit a query")
        print("4. Wait for response")
        print("5. Note any errors or slowness")
        print("\nPress Ctrl+C when done testing")

def main():
    # Configuration
    APP_URL = "http://localhost:8501"  # Change this to your Domino app URL
    
    print("=" * 50)
    print("STREAMLIT LOAD TEST COORDINATOR")
    print("=" * 50)
    
    print("\nSelect test scenario:")
    print("1. Light Load (5 tabs)")
    print("2. Medium Load (10 tabs)")
    print("3. Heavy Load (20 tabs)")
    print("4. Stress Test (30 tabs)")
    print("5. Custom")
    
    choice = input("\nEnter choice (1-5): ")
    
    tab_counts = {
        '1': 5,
        '2': 10,
        '3': 20,
        '4': 30
    }
    
    if choice in tab_counts:
        num_tabs = tab_counts[choice]
    elif choice == '5':
        num_tabs = int(input("Enter number of tabs: "))
    else:
        print("Invalid choice")
        return
    
    stagger = input("Seconds between opening tabs (default 2): ") or "2"
    stagger_seconds = float(stagger)
    
    # Update URL if needed
    custom_url = input(f"App URL (press Enter for {APP_URL}): ")
    if custom_url:
        APP_URL = custom_url
    
    # Create and run test
    tester = LoadTestCoordinator(APP_URL, num_tabs)
    
    print(f"\nStarting test in 3 seconds...")
    time.sleep(3)
    
    try:
        tester.open_tabs_gradually(stagger_seconds)
        # Keep script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nTest completed!")
        print("Check the metrics dashboard in your Streamlit app for results")

if __name__ == "__main__":
    main()