# Yantra Central — Smart City Infrastructure Command Center

Yantra Central is an AI-powered urban infrastructure health monitoring system designed to provide real-time insights into drainage systems, flood management, road health, and bridge structural integrity.

## Features

- **Command Center Dashboard**: Centralized view of all infrastructure assets with real-time status and alerts.
- **Drainage & Flood Monitoring**: 
  - Real-time water level and flow rate monitoring.
  - **AI Stress Score**: Predictive model using Random Forest to forecast drainage stress based on rainfall and soil moisture.
  - Interactive map visualization of drainage nodes and risk zones.
- **Road Health Monitoring**:
  - Tracks road condition, pothole counts, and surface quality.
  - Historical health trends and maintenance logs.
- **Bridge Health Monitoring**:
  - Structural health tracking (vibration, load factors, crack index).
  - Predictive maintenance scheduling.
- **AI Insights & Alerts**:
  - Generates predictive alerts for potential failures (e.g., flood risk, blockage detection).
  - Provides actionable repair recommendations with estimated costs and priority.

## Tech Stack

### Backend
- **Language**: Python 3.1x
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Machine Learning**: 
  - `scikit-learn` (RandomForestRegressor for stress prediction)
  - `pandas`, `numpy` for data processing
- **Database**: MongoDB (used for storing sensor readings)

### Frontend
- **Core**: HTML5, CSS3 (Custom variables, Responsive Grid), Vanilla JavaScript (ES6+)
- **Visualization**: 
  - [Chart.js](https://www.chartjs.org/) for analytics charts.
  - [Leaflet.js](https://leafletjs.com/) for interactive city maps.
- **Icons**: [Lucide](https://lucide.dev/)
- **Design**: Modern, dark-mode focused UI with glassmorphism effects.

## Project Structure

├── app/
│   ├── main.py              # Main FastAPI application & ML logic
│   ├── static/              # Frontend Assets
│   │   ├── index.html       # Single Page Application entry
│   │   ├── css/style.css    # Custom styling
│   │   └── js/dashboard.js  # Frontend logic & API integration
│   ├── models/              # Saved ML models (.pkl files)
│   └── requirements.txt     # Python dependencies for the app
├── *.pkl                    # Root level ML model artifacts (backups/training)
├── verify_db.py             # Script to verify MongoDB connection and insert test data
└── README.md                # Project Documentation

##  Setup & Installation

### Prerequisites
- Python 3.9+
- MongoDB (optional, for data ingestion features)

### 1. Install Dependencies
Navigate to the project directory and install the required Python packages.

```bash
pip install fastapi uvicorn pandas numpy scikit-learn python-dotenv pymongo
```

### 2. Run the Application
Start the FastAPI backend server.

```bash
cd app
uvicorn main:app --reload
```
*The server will start at `http://localhost:8000`*

### 3. Access the Dashboard
Open your web browser and navigate to:
**http://localhost:8000**

## Machine Learning Models

The system uses a **Random Forest Regressor** to calculate a "Stress Score" for the drainage network.
- **Inputs**: Rainfall (3-day total), Soil Moisture, Drain Water Levels.
- **Training Data**: Historical rainfall data (`e5c275eb-a4f2-4412-9677-73654e8f5f4d.csv`) and soil moisture data (`sm_Tamilnadu_2020.csv`).
- **Logic**: The model runs on startup (`app/main.py`) to train or load existing `.pkl` assets.

## API Endpoints

- **GET /**: Serves the main Dashboard.
- **GET /api/kpis**: System-wide key performance indicators.
- **GET /api/nodes**: Status of all drainage nodes.
- **GET /api/alerts**: Real-time generated alerts.
- **GET /api/stress-score**: ML-predicted stress index and history.
- **GET /api/road-health**: Mocked data for road segment conditions.
- **GET /api/bridge-health**: Mocked data for bridge structural analysis.
- **POST /reading**: Endpoint to receive sensor data (rainfall, water_level, soil_moisture).

---