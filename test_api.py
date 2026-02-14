import requests
import sys

BASE_URL = "http://localhost:8000"

def test_backend():
    print(f"Testing connectivity to {BASE_URL}...")
    try:
        # 1. Register a test user
        username = "test_debug_user"
        password = "testpassword123"
        print("1. Registering/Logging in...")
        
        # Try login first in case user exists
        login_payload = {"username": username, "password": password}
        resp = requests.post(f"{BASE_URL}/token", data=login_payload)
        
        if resp.status_code != 200:
            # Try registering
            reg_payload = {"username": username, "password": password, "role": "authority", "organization": "Debug Org"}
            resp = requests.post(f"{BASE_URL}/register", json=reg_payload)
            if resp.status_code != 200 and "already registered" not in resp.text:
                print(f"Registration failed: {resp.status_code} {resp.text}")
                return
            
            # Login again
            resp = requests.post(f"{BASE_URL}/token", data=login_payload)
            if resp.status_code != 200:
                print(f"Login failed: {resp.status_code} {resp.text}")
                return

        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("Login successful. Token obtained.")

        # 2. Check /api/users/me
        print("2. Checking /api/users/me...")
        resp = requests.get(f"{BASE_URL}/api/users/me", headers=headers, timeout=5)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Response: {resp.json()}")
        else:
            print(f"Error: {resp.text}")

        # 3. Check other APIs
        endpoints = ["/api/nodes", "/api/alerts", "/api/system-status"]
        for ep in endpoints:
            print(f"Checking {ep}...")
            try:
                r = requests.get(f"{BASE_URL}{ep}", headers=headers, timeout=5)
                print(f"{ep}: {r.status_code} (Length: {len(r.content)})")
            except requests.Timeout:
                print(f"{ep}: TIMED OUT")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_backend()
