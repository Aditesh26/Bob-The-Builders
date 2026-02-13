import requests
import json
import time

def check_insights():
    print("Fetching AI Insights (this may take a few seconds)...")
    start = time.time()
    try:
        r = requests.get("http://localhost:8000/api/insights", timeout=15)
        duration = time.time() - start
        print(f"Status: {r.status_code} (took {duration:.2f}s)")
        
        if r.status_code == 200:
            insights = r.json()
            print(f"Insights received: {len(insights)}")
            print(json.dumps(insights, indent=2))
        else:
            print(f"Error: {r.text}")

    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    check_insights()
