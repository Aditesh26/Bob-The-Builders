import requests
import time
import concurrent.futures

def check_endpoint(url):
    start = time.time()
    try:
        r = requests.get(url, timeout=10)
        duration = time.time() - start
        print(f"[{r.status_code}] {url} took {duration:.2f}s")
        return r.status_code
    except Exception as e:
        duration = time.time() - start
        print(f"[ERR] {url} failed after {duration:.2f}s: {e}")
        return None

def test_all_endpoints():
    base_url = "http://localhost:8000"
    endpoints = [
        "/api/kpis",
        "/api/sensors/rainfall",
        "/api/sensors/water_level",
        "/api/sensors/soil_moisture",
        "/api/nodes",
        "/api/zones",
        "/api/alerts",
        "/api/recommendations",
        "/api/insights",
        "/api/system-status",
        "/api/stress-score",
        "/api/stress-score/history",
        "/api/road-health",
        "/api/bridge-health"
    ]
    
    print("Testing all endpoints concurrently (simulation browser load)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_endpoint, f"{base_url}{ep}"): ep for ep in endpoints}
        for future in concurrent.futures.as_completed(futures):
            ep = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Endpoint {ep} generated an exception: {e}")

if __name__ == "__main__":
    time.sleep(2) # Wait for server reload
    test_all_endpoints()
