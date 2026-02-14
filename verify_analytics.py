import requests
import json

BASE_URL = "http://localhost:8000"

def check_endpoint(endpoint):
    try:
        url = f"{BASE_URL}{endpoint}"
        print(f"Checking {url}...")
        resp = requests.get(url, timeout=5)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                print(f"Data type: List (Length: {len(data)})")
                if len(data) > 0:
                    print("Sample item:", data[0])
            elif isinstance(data, dict):
                print("Data type: Dict")
                print("Keys:", list(data.keys()))
            else:
                print("Data:", data)
            return True
        else:
            print("Error response:", resp.text)
            return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def verify_analytics():
    print("--- Verifying Analytics Endpoints ---")
    endpoints = [
        "/api/sensors/rainfall",
        "/api/sensors/water_level",
        "/api/sensors/soil_moisture"
    ]
    
    all_passed = True
    for ep in endpoints:
        if not check_endpoint(ep):
            all_passed = False
            
    if all_passed:
        print("\nAll analytics endpoints seem to be working.")
    else:
        print("\nSome endpoints failed.")

if __name__ == "__main__":
    verify_analytics()
