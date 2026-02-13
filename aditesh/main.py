from fastapi import FastAPI
from database import readings_collection
from datetime import datetime

app = FastAPI()

@app.post("/reading")
def add_reading(rainfall: float, water_level: float, soil_moisture: float):

    data = {
        "rainfall": rainfall,
        "water_level": water_level,
        "soil_moisture": soil_moisture,
        "timestamp": datetime.utcnow()
    }

    result = readings_collection.insert_one(data)

    # return safe JSON
    return {
        "message": "Reading stored",
        "id": str(result.inserted_id)
    }
