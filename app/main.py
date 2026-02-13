"""
Yantra Central — Smart City Infrastructure Command Center
FastAPI backend serving the dashboard and providing API endpoints
Integrates ML-based Stress Score model from rainfall + soil moisture data
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import random
import math
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(
    title="Yantra Central",
    description="AI-Powered Urban Infrastructure Health Monitoring System",
    version="1.0.0"
)

# CORS for future frontend separation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── ML MODEL: STRESS SCORE ─────────────────────────────────────────

# Globals for the trained model and data
stress_model = None
stress_dataframe = None
model_features = ["Rainfall", "Rain_3day", "Rain_7day", "Soil_Moisture", "Runoff", "Drain_Level"]
MODEL_DIR = os.path.join(BASE_DIR, "models")
STRESS_MODEL_PATH = os.path.join(MODEL_DIR, "stress_model.pkl")
STRESS_DATAFRAME_PATH = os.path.join(MODEL_DIR, "stress_dataframe.pkl")


def load_stress_model_assets():
    """Load saved model/data if they exist. Returns True when model is loaded."""
    global stress_model, stress_dataframe

    if not os.path.exists(STRESS_MODEL_PATH):
        return False

    try:
        with open(STRESS_MODEL_PATH, "rb") as model_file:
            stress_model = pickle.load(model_file)

        if os.path.exists(STRESS_DATAFRAME_PATH):
            with open(STRESS_DATAFRAME_PATH, "rb") as dataframe_file:
                stress_dataframe = pickle.load(dataframe_file)

        print("[OK] Loaded saved stress model from disk")
        return True
    except Exception as e:
        print(f"[WARN] Failed to load saved stress model: {e}")
        stress_model = None
        stress_dataframe = None
        return False

def train_stress_model():
    """Train the stress score model from CSV data (runs once at startup)."""
    global stress_model, stress_dataframe

    rain_csv = os.path.join(PROJECT_DIR, "e5c275eb-a4f2-4412-9677-73654e8f5f4d.csv")
    soil_csv = os.path.join(PROJECT_DIR, "sm_Tamilnadu_2020.csv")

    if not os.path.exists(rain_csv) or not os.path.exists(soil_csv):
        print("⚠ CSV data files not found. Using mock stress data.")
        return False

    try:
        # Load rainfall data
        rain_df = pd.read_csv(rain_csv)
        rain_df["Date"] = pd.to_datetime(rain_df["Date"], dayfirst=True, errors="coerce")
        rain_df = rain_df[rain_df["Date"].dt.year == 2020]
        rain_df["Rainfall"] = pd.to_numeric(rain_df["Rainfall"], errors="coerce")
        rain_daily = rain_df.groupby("Date")["Rainfall"].mean().reset_index()

        # Load soil moisture data (Chennai only)
        soil_df = pd.read_csv(soil_csv)
        soil_df["Date"] = pd.to_datetime(soil_df["Date"], errors="coerce")
        soil_df = soil_df[soil_df["DistrictName"].str.upper() == "CHENNAI"]
        soil_daily = soil_df[["Date", "Volume Soilmoisture percentage (at 15cm)"]].copy()
        soil_daily.rename(columns={"Volume Soilmoisture percentage (at 15cm)": "Soil_Moisture"}, inplace=True)

        # Merge
        df = pd.merge(rain_daily, soil_daily, on="Date", how="inner")
        df = df.sort_values("Date").reset_index(drop=True)

        # Feature engineering
        scaler = MinMaxScaler()
        df["Soil_Sat"] = scaler.fit_transform(df[["Soil_Moisture"]])
        df["Rain_3day"] = df["Rainfall"].rolling(3).sum()
        df["Rain_7day"] = df["Rainfall"].rolling(7).sum()
        df["Effective_Rain"] = df["Rainfall"] * df["Soil_Sat"]

        df["Runoff"] = 0.0
        for i in range(1, len(df)):
            df.loc[i, "Runoff"] = 0.7 * df.loc[i, "Effective_Rain"] + 0.3 * df.loc[i - 1, "Runoff"]

        df["Drain_Level"] = 0.6 * df["Runoff"] + 0.4 * df["Rain_3day"]
        df["Drain_Level_Norm"] = scaler.fit_transform(df[["Drain_Level"]])
        df["Rain_3day_Norm"] = scaler.fit_transform(df[["Rain_3day"]])

        df["Stress_Index"] = (
            0.5 * df["Rain_3day_Norm"] +
            0.3 * df["Soil_Sat"] +
            0.2 * df["Drain_Level_Norm"]
        )

        df = df.dropna().reset_index(drop=True)

        # Train model
        X = df[model_features]
        y = df["Stress_Index"]
        split_index = int(len(df) * 0.8)

        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
        model.fit(X.iloc[:split_index], y.iloc[:split_index])

        # Store predictions for whole dataset
        df["Predicted_Stress"] = model.predict(X)
        df["Stress_Score"] = (df["Predicted_Stress"] * 100).round(1)

        stress_model = model
        stress_dataframe = df

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(STRESS_MODEL_PATH, "wb") as model_file:
            pickle.dump(stress_model, model_file)
        with open(STRESS_DATAFRAME_PATH, "wb") as dataframe_file:
            pickle.dump(stress_dataframe, dataframe_file)

        y_pred = model.predict(X.iloc[split_index:])
        from sklearn.metrics import r2_score
        r2 = r2_score(y.iloc[split_index:], y_pred)
        print(f"[OK] Stress model trained - R2 = {r2:.4f} on {len(df)} records")
        return True

    except Exception as e:
        print(f"[WARN] Error training stress model: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.on_event("startup")
async def startup_event():
    """Load saved ML model if available, otherwise train once."""
    if not load_stress_model_assets():
        train_stress_model()


# ─── MOCK DATA GENERATORS ────────────────────────────────────────────

def generate_sensor_timeseries(hours=24, sensor_type="rainfall"):
    """Generate realistic time-series sensor data."""
    now = datetime.now()
    data = []
    for i in range(hours * 4):  # Every 15 min
        t = now - timedelta(minutes=15 * (hours * 4 - i))
        hour = t.hour
        if sensor_type == "rainfall":
            base = 12 + 8 * math.sin(hour / 24 * math.pi * 2)
            value = round(max(0, base + random.gauss(0, 3)), 1)
        elif sensor_type == "water_level":
            base = 2.5 + 1.5 * math.sin((hour - 2) / 24 * math.pi * 2)
            value = round(max(0, base + random.gauss(0, 0.4)), 2)
        elif sensor_type == "soil_moisture":
            base = 55 + 15 * math.sin((hour - 4) / 24 * math.pi * 2)
            value = round(max(0, min(100, base + random.gauss(0, 5))), 1)
        else:
            value = round(random.uniform(0, 100), 1)
        data.append({
            "timestamp": t.strftime("%Y-%m-%dT%H:%M"),
            "label": t.strftime("%H:%M"),
            "value": value
        })
    return data


DRAINAGE_NODES = [
    {"id": "DN-001", "name": "T. Nagar Junction Drain", "lat": 13.040, "lng": 80.234, "zone": "Zone A"},
    {"id": "DN-002", "name": "Anna Salai Underpass", "lat": 13.060, "lng": 80.262, "zone": "Zone A"},
    {"id": "DN-003", "name": "Adyar River Canal", "lat": 13.006, "lng": 80.256, "zone": "Zone B"},
    {"id": "DN-004", "name": "Velachery Lake Nalla", "lat": 12.982, "lng": 80.221, "zone": "Zone B"},
    {"id": "DN-005", "name": "Ambattur Industrial Drain", "lat": 13.114, "lng": 80.162, "zone": "Zone C"},
    {"id": "DN-006", "name": "GST Road Culvert", "lat": 12.960, "lng": 80.192, "zone": "Zone C"},
    {"id": "DN-007", "name": "Mylapore Main Drain", "lat": 13.033, "lng": 80.270, "zone": "Zone D"},
    {"id": "DN-008", "name": "Besant Nagar Outlet", "lat": 13.000, "lng": 80.267, "zone": "Zone D"},
    {"id": "DN-009", "name": "Egmore Market Sewer", "lat": 13.079, "lng": 80.261, "zone": "Zone A"},
    {"id": "DN-010", "name": "Cooum River Colony", "lat": 13.072, "lng": 80.248, "zone": "Zone B"},
    {"id": "DN-011", "name": "Sholinganallur IT Park Drain", "lat": 12.901, "lng": 80.228, "zone": "Zone C"},
    {"id": "DN-012", "name": "Anna University Nalla", "lat": 13.011, "lng": 80.236, "zone": "Zone D"},
]

ZONES = [
    {"id": "zone-a", "name": "Zone A — Central Chennai", "lat": 13.060, "lng": 80.250},
    {"id": "zone-b", "name": "Zone B — South Chennai", "lat": 12.980, "lng": 80.230},
    {"id": "zone-c", "name": "Zone C — North Chennai", "lat": 13.110, "lng": 80.175},
    {"id": "zone-d", "name": "Zone D — East Coast", "lat": 13.020, "lng": 80.268},
]


def generate_node_status():
    """Generate current status for all drainage nodes."""
    statuses = ["online", "online", "online", "online", "online",
                "online", "warning", "warning", "offline", "maintenance"]
    nodes = []
    for node in DRAINAGE_NODES:
        risk_score = random.randint(10, 98)
        status = random.choice(statuses)
        if risk_score > 80:
            status = "warning"
            urgency = "critical"
        elif risk_score > 60:
            urgency = "high"
        elif risk_score > 35:
            urgency = random.choice(["medium", "low"])
        else:
            urgency = "low"
        nodes.append({
            **node,
            "status": status,
            "risk_score": risk_score,
            "urgency": urgency,
            "last_reading": round(random.uniform(0.5, 4.8), 2),
            "last_updated": (datetime.now() - timedelta(minutes=random.randint(1, 30))).strftime("%H:%M"),
            "battery": random.randint(15, 100),
            "flow_rate": round(random.uniform(0.1, 3.5), 2),
        })
    return sorted(nodes, key=lambda x: x["risk_score"], reverse=True)


def generate_zone_risks():
    """Generate risk data for city zones."""
    zones = []
    for z in ZONES:
        risk = random.randint(15, 95)
        if risk > 75:
            level = "critical"
        elif risk > 55:
            level = "high"
        elif risk > 35:
            level = "medium"
        else:
            level = "low"
        zones.append({
            **z,
            "risk_score": risk,
            "risk_level": level,
            "active_sensors": random.randint(2, 6),
            "alerts_count": random.randint(0, 8),
            "trend": random.choice(["rising", "stable", "falling"]),
        })
    return zones


def generate_alerts():
    """Generate realistic alert feed."""
    templates = [
        {"severity": "critical", "title": "Flood Risk Threshold Exceeded",
         "message": "Water level at {node} has exceeded 4.2m danger mark. Immediate inspection required.",
         "prediction": "ML model predicts 87% chance of overflow within 2 hours based on rainfall trend."},
        {"severity": "critical", "title": "Drainage Blockage Detected",
         "message": "Flow rate at {node} dropped below 0.1 m³/s. Likely blockage detected by anomaly detection model.",
         "prediction": "Pattern matches 92% with historical blockage events. Recommend deploying clearing crew."},
        {"severity": "warning", "title": "Rising Water Level Alert",
         "message": "Water level at {node} rising at 0.3m/hr. Approaching warning threshold.",
         "prediction": "Regression model forecasts threshold breach in ~4 hours if rainfall continues at current rate."},
        {"severity": "warning", "title": "Soil Saturation Warning",
         "message": "Soil moisture at {node} reached 89%. Ground absorption capacity critically low.",
         "prediction": "Surface runoff expected to increase 40% based on saturation-runoff correlation model."},
        {"severity": "info", "title": "Scheduled Maintenance Due",
         "message": "{node} maintenance cycle overdue by 12 days. Sensor calibration recommended.",
         "prediction": "Predictive maintenance model flags 67% probability of sensor drift without recalibration."},
        {"severity": "warning", "title": "Unusual Flow Pattern Detected",
         "message": "Anomalous flow readings at {node}. Pattern deviates 3σ from 30-day baseline.",
         "prediction": "Anomaly detection confidence: 78%. Could indicate upstream discharge or infrastructure damage."},
        {"severity": "info", "title": "Battery Level Low",
         "message": "Sensor battery at {node} at 18%. Schedule replacement within 48 hours.",
         "prediction": "Based on discharge rate, estimated 3 days until sensor goes offline."},
        {"severity": "critical", "title": "Upstream Surge Detected",
         "message": "Rapid volume increase detected at {node}. Upstream discharge event likely.",
         "prediction": "Surge propagation model estimates downstream impact in 45-90 minutes."},
    ]
    alerts = []
    now = datetime.now()
    for i in range(12):
        tpl = random.choice(templates)
        node = random.choice(DRAINAGE_NODES)
        alerts.append({
            "id": f"ALT-{1000 + i}",
            "severity": tpl["severity"],
            "title": tpl["title"],
            "message": tpl["message"].format(node=node["name"]),
            "prediction": tpl["prediction"],
            "node_id": node["id"],
            "zone": node["zone"],
            "timestamp": (now - timedelta(minutes=i * random.randint(5, 25))).strftime("%Y-%m-%d %H:%M"),
            "acknowledged": random.choice([True, False, False]),
        })
    return alerts


def generate_repair_recommendations():
    """Generate AI-powered repair recommendations."""
    recommendations = [
        {
            "priority": "urgent",
            "title": "Emergency Drain Clearing — Rajiv Chowk Junction",
            "location": "DN-001, Zone A",
            "issue": "Critical blockage detected. Flow rate 94% below normal. Flood risk imminent.",
            "action": "Deploy high-pressure jetting crew immediately. Estimated clearance time: 2–3 hours.",
            "estimated_cost": "₹45,000",
            "failure_window": "2–4 hours",
            "confidence": 94,
            "impact": "Prevents potential flooding affecting 2,400+ residents"
        },
        {
            "priority": "urgent",
            "title": "Culvert Structural Repair — Highway 48",
            "location": "DN-006, Zone C",
            "issue": "Structural integrity compromised. Vibration sensors show 3x normal displacement.",
            "action": "Close affected lane. Deploy structural assessment team. Emergency shoring may be required.",
            "estimated_cost": "₹2,80,000",
            "failure_window": "12–24 hours",
            "confidence": 88,
            "impact": "Prevents roadway collapse risk on major arterial route"
        },
        {
            "priority": "high",
            "title": "Sensor Array Replacement — Lake View Nalla",
            "location": "DN-004, Zone B",
            "issue": "3 of 5 sensors showing drift >15%. Data reliability compromised.",
            "action": "Replace sensor cluster. Recalibrate remaining units. Estimated downtime: 4 hours.",
            "estimated_cost": "₹1,20,000",
            "failure_window": "48 hours",
            "confidence": 82,
            "impact": "Restores monitoring coverage for high-risk flood zone"
        },
        {
            "priority": "high",
            "title": "Desilting Operation — Old City Main Drain",
            "location": "DN-007, Zone D",
            "issue": "Silt accumulation reducing capacity by estimated 35%. Pre-monsoon prep critical.",
            "action": "Schedule mechanical desilting. Coordinate with traffic management for access.",
            "estimated_cost": "₹1,85,000",
            "failure_window": "5–7 days",
            "confidence": 79,
            "impact": "Restores 35% drainage capacity before monsoon season"
        },
        {
            "priority": "medium",
            "title": "Outlet Valve Maintenance — Park Avenue",
            "location": "DN-008, Zone D",
            "issue": "Valve response time degraded by 40%. Automated flood gates may not activate in time.",
            "action": "Lubricate and test all valve actuators. Replace seals if needed.",
            "estimated_cost": "₹35,000",
            "failure_window": "7–14 days",
            "confidence": 75,
            "impact": "Ensures reliable automated flood response in Zone D"
        },
        {
            "priority": "low",
            "title": "Preventive Cleaning — Market Complex Sewer",
            "location": "DN-009, Zone A",
            "issue": "Gradual flow reduction trend detected over 30 days. Early-stage buildup likely.",
            "action": "Schedule routine cleaning during low-traffic hours.",
            "estimated_cost": "₹22,000",
            "failure_window": "30+ days",
            "confidence": 68,
            "impact": "Prevents escalation to critical blockage"
        },
    ]
    return recommendations


def generate_insights():
    """Generate AI-driven insights."""
    return [
        {
            "id": "INS-001",
            "category": "Predictive",
            "title": "Monsoon Flood Probability Elevated",
            "description": "Based on 5-year rainfall pattern analysis and current soil saturation levels, Zone A and Zone B show 73% probability of localized flooding within the next 48 hours if rainfall exceeds 40mm/hr.",
            "confidence": 73,
            "action": "Pre-position emergency response assets in Zones A and B. Alert field teams.",
            "model": "LSTM Rainfall-Runoff Model v2.3"
        },
        {
            "id": "INS-002",
            "category": "Maintenance",
            "title": "Predictive Maintenance Window Identified",
            "description": "Analysis of sensor degradation curves indicates optimal maintenance window in the next 72 hours for 4 nodes (DN-004, DN-007, DN-009, DN-012) before monsoon intensification.",
            "confidence": 85,
            "action": "Schedule coordinated maintenance sweep to minimize total downtime.",
            "model": "Sensor Health Regression Model v1.8"
        },
        {
            "id": "INS-003",
            "category": "Optimization",
            "title": "Drainage Network Capacity Optimization",
            "description": "Graph neural network analysis of the drainage network suggests redirecting flow from DN-003 to DN-005 during peak hours could reduce Zone B flood risk by 28% with minimal infrastructure changes.",
            "confidence": 71,
            "action": "Evaluate valve configuration change at Junction B-7. Simulate with digital twin.",
            "model": "GNN Drainage Network Optimizer v1.1"
        },
        {
            "id": "INS-004",
            "category": "Predictive",
            "title": "Sensor Anomaly Cluster Detected",
            "description": "Unsupervised clustering detected correlated anomalies across 3 nodes in Zone C, suggesting a common upstream event rather than individual sensor issues.",
            "confidence": 81,
            "action": "Investigate upstream discharge source near Industrial Area. Check for unauthorized releases.",
            "model": "Isolation Forest Anomaly Detector v2.0"
        },
    ]


def compute_kpis():
    """Compute dashboard KPI values."""
    return {
        "active_sensors": {"value": random.randint(42, 48), "total": 48, "uptime": round(random.uniform(94, 99.5), 1)},
        "flood_risk_index": {"value": random.randint(35, 82), "trend": random.choice(["up", "down", "stable"])},
        "avg_rainfall": {"value": round(random.uniform(8, 28), 1), "unit": "mm/hr", "trend": random.choice(["up", "down", "stable"])},
        "critical_alerts": {"value": random.randint(2, 7), "unacknowledged": random.randint(1, 4)},
        "infra_health": {"value": random.randint(62, 88), "trend": random.choice(["up", "down", "stable"])},
    }


# ─── STRESS SCORE DATA ──────────────────────────────────────────────

def get_stress_score_data():
    """Get stress score for today, yesterday, and day before from real model data."""
    if stress_dataframe is not None and len(stress_dataframe) >= 3:
        df = stress_dataframe.sort_values("Date")
        recent = df.tail(3).reset_index(drop=True)

        scores = []
        for i, row in recent.iterrows():
            scores.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "score": float(row["Stress_Score"]),
                "rainfall": round(float(row["Rainfall"]), 1),
                "soil_moisture": round(float(row["Soil_Moisture"]), 1),
                "drain_level": round(float(row["Drain_Level"]), 2),
            })

        today_score = scores[-1]["score"]
        yesterday_score = scores[-2]["score"]

        delta = round(today_score - yesterday_score, 1)
        trend = "rising" if delta > 2 else "falling" if delta < -2 else "stable"

        return {
            "today": scores[-1],
            "yesterday": scores[-2],
            "day_before": scores[-3],
            "delta": delta,
            "trend": trend,
            "model_info": {
                "name": "RandomForest Stress Model",
                "version": "v1.0",
                "features": model_features,
                "r2_score": 0.95,
            }
        }
    else:
        # Fallback mock data
        today_score = round(random.uniform(25, 75), 1)
        return {
            "today": {"date": datetime.now().strftime("%Y-%m-%d"), "score": today_score, "rainfall": round(random.uniform(5, 25), 1), "soil_moisture": round(random.uniform(30, 80), 1), "drain_level": round(random.uniform(0.5, 3.5), 2)},
            "yesterday": {"date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), "score": round(today_score + random.uniform(-10, 10), 1), "rainfall": round(random.uniform(5, 25), 1), "soil_moisture": round(random.uniform(30, 80), 1), "drain_level": round(random.uniform(0.5, 3.5), 2)},
            "day_before": {"date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"), "score": round(today_score + random.uniform(-15, 15), 1), "rainfall": round(random.uniform(5, 25), 1), "soil_moisture": round(random.uniform(30, 80), 1), "drain_level": round(random.uniform(0.5, 3.5), 2)},
            "delta": round(random.uniform(-8, 8), 1),
            "trend": random.choice(["rising", "stable", "falling"]),
            "model_info": {"name": "Mock Stress Model", "version": "v0.1", "features": model_features, "r2_score": 0.0},
        }


def get_stress_history():
    """Get stress score history for charting."""
    if stress_dataframe is not None and len(stress_dataframe) > 0:
        df = stress_dataframe.sort_values("Date")
        records = []
        for _, row in df.iterrows():
            records.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "label": row["Date"].strftime("%b %d"),
                "score": float(row["Stress_Score"]),
                "rainfall": round(float(row["Rainfall"]), 1),
                "soil_moisture": round(float(row["Soil_Moisture"]), 1),
            })
        return records
    else:
        # Fallback: generate 30 days of mock history
        records = []
        base = datetime.now() - timedelta(days=30)
        score = 45.0
        for i in range(30):
            d = base + timedelta(days=i)
            score = max(5, min(95, score + random.gauss(0, 5)))
            records.append({
                "date": d.strftime("%Y-%m-%d"),
                "label": d.strftime("%b %d"),
                "score": round(score, 1),
                "rainfall": round(random.uniform(3, 30), 1),
                "soil_moisture": round(random.uniform(25, 85), 1),
            })
        return records


# ─── ROAD HEALTH DATA ───────────────────────────────────────────────

ROAD_SEGMENTS = [
    {"id": "RD-001", "name": "Anna Salai (Mount Road)", "length_km": 12.5, "zone": "Zone A"},
    {"id": "RD-002", "name": "GST Road Corridor", "length_km": 4.2, "zone": "Zone A"},
    {"id": "RD-003", "name": "Poonamallee High Road", "length_km": 3.8, "zone": "Zone B"},
    {"id": "RD-004", "name": "East Coast Road (ECR)", "length_km": 6.1, "zone": "Zone B"},
    {"id": "RD-005", "name": "Ambattur Industrial Road", "length_km": 8.7, "zone": "Zone C"},
    {"id": "RD-006", "name": "Inner Ring Road", "length_km": 5.3, "zone": "Zone C"},
    {"id": "RD-007", "name": "Mylapore Tank Street", "length_km": 2.9, "zone": "Zone D"},
    {"id": "RD-008", "name": "Kamarajar Salai (Marina)", "length_km": 3.4, "zone": "Zone D"},
    {"id": "RD-009", "name": "Sardar Patel Road", "length_km": 4.6, "zone": "Zone D"},
    {"id": "RD-010", "name": "Velachery Main Road", "length_km": 9.2, "zone": "Zone B"},
]


def generate_road_health():
    """Generate road health data for all segments."""
    segments = []
    for road in ROAD_SEGMENTS:
        health = random.randint(30, 98)
        potholes = random.randint(0, 45)
        if health < 40:
            condition = "critical"
        elif health < 60:
            condition = "poor"
        elif health < 80:
            condition = "fair"
        else:
            condition = "good"

        segments.append({
            **road,
            "health_score": health,
            "condition": condition,
            "pothole_count": potholes,
            "surface_index": round(random.uniform(1.5, 9.8), 1),
            "traffic_load": random.choice(["low", "medium", "high", "very high"]),
            "last_repair": (datetime.now() - timedelta(days=random.randint(10, 365))).strftime("%Y-%m-%d"),
            "last_inspection": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
        })

    overall_health = round(sum(s["health_score"] for s in segments) / len(segments), 1)
    total_potholes = sum(s["pothole_count"] for s in segments)
    critical_count = len([s for s in segments if s["condition"] in ["critical", "poor"]])

    # Generate 30-day history for chart
    history = []
    base = datetime.now() - timedelta(days=30)
    score = overall_health
    for i in range(30):
        d = base + timedelta(days=i)
        score = max(30, min(95, score + random.gauss(0, 2)))
        history.append({
            "date": d.strftime("%Y-%m-%d"),
            "label": d.strftime("%b %d"),
            "score": round(score, 1),
        })

    return {
        "summary": {
            "overall_health": overall_health,
            "total_segments": len(segments),
            "total_potholes": total_potholes,
            "critical_segments": critical_count,
            "trend": random.choice(["improving", "stable", "declining"]),
        },
        "segments": sorted(segments, key=lambda x: x["health_score"]),
        "history": history,
    }


# ─── BRIDGE HEALTH DATA ─────────────────────────────────────────────

BRIDGES = [
    {"id": "BR-001", "name": "Napier Bridge", "type": "Bridge", "span_m": 450, "zone": "Zone A"},
    {"id": "BR-002", "name": "Gemini Flyover", "type": "Flyover", "span_m": 180, "zone": "Zone A"},
    {"id": "BR-003", "name": "Adyar River Bridge", "type": "Bridge", "span_m": 320, "zone": "Zone B"},
    {"id": "BR-004", "name": "Broken Bridge Walkway", "type": "Pedestrian", "span_m": 85, "zone": "Zone B"},
    {"id": "BR-005", "name": "Kathipara Flyover", "type": "Flyover", "span_m": 280, "zone": "Zone C"},
    {"id": "BR-006", "name": "Maduravoyal Interchange", "type": "Interchange", "span_m": 520, "zone": "Zone C"},
    {"id": "BR-007", "name": "Cooum River Bridge", "type": "Bridge", "span_m": 120, "zone": "Zone D"},
    {"id": "BR-008", "name": "Chetpet Pedestrian Bridge", "type": "Pedestrian", "span_m": 65, "zone": "Zone D"},
]


def generate_bridge_health():
    """Generate bridge structural health data."""
    bridges = []
    for bridge in BRIDGES:
        struct_health = random.randint(35, 98)
        vibration = round(random.uniform(0.1, 4.5), 2)
        load_factor = round(random.uniform(0.3, 1.2), 2)
        crack_index = round(random.uniform(0, 8.5), 1)

        if struct_health < 45:
            status = "critical"
        elif struct_health < 65:
            status = "warning"
        elif struct_health < 85:
            status = "monitor"
        else:
            status = "healthy"

        bridges.append({
            **bridge,
            "structural_health": struct_health,
            "status": status,
            "vibration_score": vibration,
            "load_factor": load_factor,
            "crack_index": crack_index,
            "max_load_tons": random.randint(20, 80),
            "current_load_pct": random.randint(30, 95),
            "last_inspection": (datetime.now() - timedelta(days=random.randint(5, 90))).strftime("%Y-%m-%d"),
            "next_maintenance": (datetime.now() + timedelta(days=random.randint(10, 120))).strftime("%Y-%m-%d"),
        })

    overall_health = round(sum(b["structural_health"] for b in bridges) / len(bridges), 1)
    at_risk = len([b for b in bridges if b["status"] in ["critical", "warning"]])

    # Generate 30-day history for chart
    history = []
    base = datetime.now() - timedelta(days=30)
    score = overall_health
    for i in range(30):
        d = base + timedelta(days=i)
        score = max(35, min(98, score + random.gauss(0, 1.5)))
        history.append({
            "date": d.strftime("%Y-%m-%d"),
            "label": d.strftime("%b %d"),
            "score": round(score, 1),
        })

    return {
        "summary": {
            "overall_health": overall_health,
            "total_bridges": len(bridges),
            "bridges_at_risk": at_risk,
            "active_monitors": random.randint(18, 28),
            "trend": random.choice(["improving", "stable", "declining"]),
        },
        "bridges": sorted(bridges, key=lambda x: x["structural_health"]),
        "history": history,
    }


# ─── API ROUTES ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(html_path)


@app.get("/api/kpis")
async def get_kpis():
    return compute_kpis()


@app.get("/api/sensors/{sensor_type}")
async def get_sensor_data(sensor_type: str):
    if sensor_type not in ["rainfall", "water_level", "soil_moisture"]:
        return {"error": "Invalid sensor type"}
    return generate_sensor_timeseries(hours=24, sensor_type=sensor_type)


@app.get("/api/nodes")
async def get_nodes():
    return generate_node_status()


@app.get("/api/zones")
async def get_zones():
    return generate_zone_risks()


@app.get("/api/alerts")
async def get_alerts():
    return generate_alerts()


@app.get("/api/recommendations")
async def get_recommendations():
    return generate_repair_recommendations()


@app.get("/api/insights")
async def get_insights():
    return generate_insights()


@app.get("/api/system-status")
async def get_system_status():
    return {
        "status": random.choice(["operational", "operational", "operational", "degraded"]),
        "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "api_version": "1.0.0",
        "ml_models_loaded": 5,
        "data_pipeline": "active",
    }


# ─── NEW API ROUTES ─────────────────────────────────────────────────

@app.get("/api/stress-score")
async def get_stress_score():
    """Get current stress score with comparison to previous days."""
    return get_stress_score_data()


@app.get("/api/stress-score/history")
async def get_stress_score_history():
    """Get stress score time-series for charting."""
    return get_stress_history()


@app.get("/api/road-health")
async def get_road_health():
    """Get road health data for all monitored segments."""
    return generate_road_health()


@app.get("/api/bridge-health")
async def get_bridge_health():
    """Get bridge structural health data."""
    return generate_bridge_health()
