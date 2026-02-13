import pymongo
from datetime import datetime
import random

# Connect
client = pymongo.MongoClient("mongodb+srv://aditeshpatro_db_user:admin123@cluster0.gxftxhm.mongodb.net/")
db = client["smart_drain"]
collection = db["readings"]

# Create a test reading with high values to trigger a change
reading = {
    "rainfall": 55.5,       # High rainfall
    "water_level": 3.8,     # High water level
    "soil_moisture": 85.0,  # High moisture
    "timestamp": datetime.utcnow()
}

print(f"Inserting test reading: {reading}")
result = collection.insert_one(reading)
print(f"Inserted ID: {result.inserted_id}")
print("\nNow check your dashboard!")
print("1. 'Stress Score' should be high/critical.")
print("2. 'Live Sensor Data' charts should show these values at the very end.")
