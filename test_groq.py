import os
import sys
from dotenv import load_dotenv

# Enhance path to find app module
sys.path.append(os.getcwd())

load_dotenv(os.path.join(os.getcwd(), "app", ".env"))

# Mocking the function since importing main might start the server/app
import requests
import json

def get_groq_recommendations(readings, risk_score):
    """Fetch repair recommendations from Groq API based on live data."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[WARN] Groq API Key missing.")
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    You are an AI assistant for a smart city drainage system.
    Current Sensor Readings:
    Rainfall: {readings['rainfall']} mm
    Soil Moisture: {readings['soil_moisture']} %
    Water Level: {readings['water_level']} m
    Predicted Flash Flood Risk Score: {risk_score} (0-100)
    
    The risk is HIGH. Generate 1 urgent repair recommendation or maintenance action to mitigate this specific risk.
    Return ONLY a raw JSON object (no markdown formatting) with these fields:
    - priority (urgent/high/medium)
    - title (short action title)
    - location (suggest a specific node ID from DN-001 to DN-012 and Zone)
    - issue (description of the problem based on sensors)
    - action (specific repair/maintenance step)
    - estimated_cost (in ₹)
    - failure_window (time estimate)
    - confidence (0-100)
    - impact (expected outcome)
    """
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            print("Success!")
            print(response.json()["choices"][0]["message"]["content"])
        else:
            print(f"[ERR] Groq API error: {response.text}")
    except Exception as e:
        print(f"[ERR] Groq request failed: {e}")

if __name__ == "__main__":
    readings = {"rainfall": 85.0, "soil_moisture": 90.0, "water_level": 4.5}
    risk_score = 95.0
    print("Testing Groq API...")
    get_groq_recommendations(readings, risk_score)
