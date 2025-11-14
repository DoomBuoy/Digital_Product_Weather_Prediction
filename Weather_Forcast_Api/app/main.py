from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import traceback
from typing import Dict, Any
import os

# Create FastAPI app
app = FastAPI(
    title="Weather Prediction API",
    description="API for weather prediction using machine learning models",
    version="1.0.0"
)

# Global variables to store loaded models
rain_model = None
precipitation_model = None

# ====================================
# Model Loading
# ====================================

@app.on_event("startup")
async def load_models():
    """Load ML models at startup"""
    global rain_model, precipitation_model
    
    try:
        # Import the rain prediction model
        from src import multi_model_pipeline  
        rain_model = multi_model_pipeline
        print("✓ Rain prediction model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading rain prediction model: {e}")
        rain_model = None
    
    try:
        # Import the precipitation prediction model
        from src import predictor  
        precipitation_model = predictor
        print("✓ Precipitation prediction model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading precipitation prediction model: {e}")
        precipitation_model = None


# ====================================
# Helper Functions
# ====================================

def validate_date_format(date_string: str) -> datetime:
    """Validate date string in YYYY-MM-DD format"""
    try:
        return datetime.strptime(date_string, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid date format. Expected format: YYYY-MM-DD"
        )

def load_weather_data() -> pd.DataFrame:
    """Load sample weather data for prediction"""
    try:
        # Try different paths to find the weather data
        paths_to_try = [
            r'data/raw/weather_data.csv',
            r'./data/raw/weather_data.csv',
            r'../data/raw/weather_data.csv',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'weather_data.csv')
        ]
        
        # Try each path
        for path in paths_to_try:
            try:
                if os.path.exists(path):
                    weather_data = pd.read_csv(path)
                    print(f"Successfully loaded weather data from: {path}")
                    return weather_data
            except Exception:
                continue
        
        # If we get here, none of the paths worked
        raise FileNotFoundError(f"Weather data file not found. Tried paths: {paths_to_try}")
    except Exception as e:
        print(f"Error loading weather data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading weather data: {str(e)}"
        )
def format_weather_data(
    date: str ,
    weather_code: float = 51.0,
    temperature_2m_mean: float = 17.627083,
    temperature_2m_max: float = 21.3375,
    temperature_2m_min: float = 13.9375,
    apparent_temperature_mean: float = 16.776638,
    apparent_temperature_max: float = 21.033524,
    apparent_temperature_min: float = 12.8664055,
    wind_speed_10m_max: float = 18.345877,
    wind_gusts_10m_max: float = 39.239998,
    wind_direction_10m_dominant: float = 182.75,
    shortwave_radiation_sum: float = 15.82,
    et0_fao_evapotranspiration: float = 3.1182382,
    sunrise: int = 1498942856,
    sunset: int = 1498978653,
    daylight_duration: float = 43439.285,
    sunshine_duration: float = 35763.71,
    rain_sum: float = 0.1,
    precipitation_sum: float = 0.1,
    snowfall_sum: float = 0.0,
    precipitation_hours: float = 1.0,
    cloud_cover_mean: float = 47.791668,
    cloud_cover_max: float = 100.0,
    cloud_cover_min: float = 1.0,
    dew_point_2m_mean: float = 12.349667,
    dew_point_2m_max: float = 14.5205,
    dew_point_2m_min: float = 10.2205,
    et0_fao_evapotranspiration_sum: float = 3.1182382,
    relative_humidity_2m_mean: float = 74.28765,
    relative_humidity_2m_max: float = 91.40758,
    relative_humidity_2m_min: float = 55.811295,
    snowfall_water_equivalent_sum: float = 0.0,
    pressure_msl_mean: float = 1017.22076,
    pressure_msl_max: float = 1020.1,
    pressure_msl_min: float = 1014.6,
    surface_pressure_mean: float = 1011.16504,
    surface_pressure_max: float = 1014.00464,
    surface_pressure_min: float = 1008.5924,
    winddirection_10m_dominant: float = 182.75,
    wind_gusts_10m_mean: float = 24.269997,
    wind_speed_10m_mean: float = 10.986862,
    wind_gusts_10m_min: float = 10.440001,
    wind_speed_10m_min: float = 4.510787,
    wet_bulb_temperature_2m_mean: float = 14.193513,
    wet_bulb_temperature_2m_max: float = 16.12211,
    wet_bulb_temperature_2m_min: float = 12.015671,
    vapour_pressure_deficit_max: float = 1.0820118,
    soil_moisture_0_to_100cm_mean: float = 0.17118753,
    soil_moisture_0_to_7cm_mean: float = 0.17545833,
    soil_moisture_28_to_100cm_mean: float = 0.171,
    soil_moisture_7_to_28cm_mean: float = 0.17770837,
    soil_temperature_0_to_100cm_mean: float = 18.333712,
    soil_temperature_0_to_7cm_mean: float = 18.549665,
    soil_temperature_7_to_28cm_mean: float = 18.475,
    soil_temperature_28_to_100cm_mean: float = 18.306252
) -> pd.DataFrame:
    """Format input parameters as a DataFrame for model prediction
    
    All parameters have default values from a typical weather data sample.
    If no parameters are provided, a complete sample weather dataframe will be returned.
    """
    return pd.DataFrame({
        'date': [date],
        'weather_code': [weather_code],
        'temperature_2m_mean': [temperature_2m_mean],
        'temperature_2m_max': [temperature_2m_max],
        'temperature_2m_min': [temperature_2m_min],
        'apparent_temperature_mean': [apparent_temperature_mean],
        'apparent_temperature_max': [apparent_temperature_max],
        'apparent_temperature_min': [apparent_temperature_min],
        'wind_speed_10m_max': [wind_speed_10m_max],
        'wind_gusts_10m_max': [wind_gusts_10m_max],
        'wind_direction_10m_dominant': [wind_direction_10m_dominant],
        'shortwave_radiation_sum': [shortwave_radiation_sum],
        'et0_fao_evapotranspiration': [et0_fao_evapotranspiration],
        'sunrise': [sunrise],
        'sunset': [sunset],
        'daylight_duration': [daylight_duration],
        'sunshine_duration': [sunshine_duration],
        'rain_sum': [rain_sum],
        'precipitation_sum': [precipitation_sum],
        'snowfall_sum': [snowfall_sum],
        'precipitation_hours': [precipitation_hours],
        'cloud_cover_mean': [cloud_cover_mean],
        'cloud_cover_max': [cloud_cover_max],
        'cloud_cover_min': [cloud_cover_min],
        'dew_point_2m_mean': [dew_point_2m_mean],
        'dew_point_2m_max': [dew_point_2m_max],
        'dew_point_2m_min': [dew_point_2m_min],
        'et0_fao_evapotranspiration_sum': [et0_fao_evapotranspiration_sum],
        'relative_humidity_2m_mean': [relative_humidity_2m_mean],
        'relative_humidity_2m_max': [relative_humidity_2m_max],
        'relative_humidity_2m_min': [relative_humidity_2m_min],
        'snowfall_water_equivalent_sum': [snowfall_water_equivalent_sum],
        'pressure_msl_mean': [pressure_msl_mean],
        'pressure_msl_max': [pressure_msl_max],
        'pressure_msl_min': [pressure_msl_min],
        'surface_pressure_mean': [surface_pressure_mean],
        'surface_pressure_max': [surface_pressure_max],
        'surface_pressure_min': [surface_pressure_min],
        'winddirection_10m_dominant': [winddirection_10m_dominant],
        'wind_gusts_10m_mean': [wind_gusts_10m_mean],
        'wind_speed_10m_mean': [wind_speed_10m_mean],
        'wind_gusts_10m_min': [wind_gusts_10m_min],
        'wind_speed_10m_min': [wind_speed_10m_min],
        'wet_bulb_temperature_2m_mean': [wet_bulb_temperature_2m_mean],
        'wet_bulb_temperature_2m_max': [wet_bulb_temperature_2m_max],
        'wet_bulb_temperature_2m_min': [wet_bulb_temperature_2m_min],
        'vapour_pressure_deficit_max': [vapour_pressure_deficit_max],
        'soil_moisture_0_to_100cm_mean': [soil_moisture_0_to_100cm_mean],
        'soil_moisture_0_to_7cm_mean': [soil_moisture_0_to_7cm_mean],
        'soil_moisture_28_to_100cm_mean': [soil_moisture_28_to_100cm_mean],
        'soil_moisture_7_to_28cm_mean': [soil_moisture_7_to_28cm_mean],
        'soil_temperature_0_to_100cm_mean': [soil_temperature_0_to_100cm_mean],
        'soil_temperature_0_to_7cm_mean': [soil_temperature_0_to_7cm_mean],
        'soil_temperature_7_to_28cm_mean': [soil_temperature_7_to_28cm_mean],
        'soil_temperature_28_to_100cm_mean': [soil_temperature_28_to_100cm_mean]
    })
# ====================================
# API Endpoints
# ====================================

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with project documentation"""
    return {
        "project": "Weather Prediction API",
        "objectives": "Predict rain and precipitation using advanced machine learning models.",
        "endpoints": {
            "/": "Project documentation and API overview",
            "/health/": "Health check endpoint",
            "/predict/rain/": "Predict if it will rain in exactly 7 days from a given date",
            "/predict/precipitation/fall/": "Predict cumulated precipitation for the next 3 days from a given date"
        },
        "expected_input_columns": [
            "date", "weather_code", "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "sunrise", "sunset", "daylight_duration", "sunshine_duration", "rain_sum", "precipitation_sum", "snowfall_sum", "precipitation_hours", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min", "dew_point_2m_mean", "dew_point_2m_max", "dew_point_2m_min", "et0_fao_evapotranspiration_sum", "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min", "snowfall_water_equivalent_sum", "pressure_msl_mean", "pressure_msl_max", "pressure_msl_min", "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min", "winddirection_10m_dominant", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min", "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean", "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_7_to_28cm_mean", "soil_temperature_28_to_100cm_mean"
        ],
        "output_format": {
            "rain_prediction": {
                "input_date": "2023-01-01",
                "prediction": {
                    "date": "2023-01-08",
                    "will_rain": True
                }
            },
            "precipitation_prediction": {
                "input_date": "2023-01-01",
                "prediction": {
                    "start_date": "2023-01-02",
                    "end_date": "2023-01-04",
                    "precipitation_fall": "28.2"
                }
            }
        }
    }

@app.get('/health', status_code=200)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Welcome to the Weather Prediction API! The service is running smoothly.",
        "models_loaded": {
            "rain_model": rain_model is not None,
            "precipitation_model": precipitation_model is not None
        }
    }

@app.get("/test/")
async def debug_info():
    """Debugging endpoint to diagnose issues"""
    results = {}
    
    # Check if data file exists
    try:
        data_paths = [
            r'data/raw/weather_data.csv',
            r'./data/raw/weather_data.csv',
            r'../data/raw/weather_data.csv'
        ]
        results["data_paths"] = {
            path: os.path.exists(path) for path in data_paths
        }
        results["current_dir"] = os.getcwd()
    except Exception as e:
        results["path_check_error"] = str(e)
    
    # Test loading data
    try:
        weather_data = load_weather_data()
        results["data_loaded"] = True
        results["data_shape"] = weather_data.shape
        results["data_columns"] = list(weather_data.columns)
        results["data_dtypes"] = {col: str(dtype) for col, dtype in weather_data.dtypes.items()}
    except Exception as e:
        results["data_load_error"] = str(e)
    
    # Test models
    if rain_model is not None:
        try:
            # Test with minimal data
            sample = pd.read_csv(r'data/raw/weather_data.csv').head(10)
            pred = rain_model.predict(sample, voting_method='majority')
            results["rain_model_test"] = "Success"
            results["rain_model_prediction"] = pred.tolist()
        except Exception as e:
            results["rain_model_test_error"] = str(e)
            results["rain_model_test_traceback"] = traceback.format_exc()
    else:
        results["rain_model"] = "Not loaded"
            
    return results

##############################################################################################################################################################################

@app.get("/predict/rain/")
async def predict_rain(
    date: str = Query(None, description="Date in YYYY-MM-DD format (REQUIRED)"),
    # Weather code
    weather_code: float = Query(None, description="Weather code"),
    # Temperature parameters
    temperature_mean: float = Query(None, description="Mean temperature in °C", alias="temperature_2m_mean"),
    temperature_max: float = Query(None, description="Maximum temperature in °C", alias="temperature_2m_max"),
    temperature_min: float = Query(None, description="Minimum temperature in °C", alias="temperature_2m_min"),
    # Apparent temperature parameters
    apparent_temperature_mean: float = Query(None, description="Mean apparent temperature in °C"),
    apparent_temperature_max: float = Query(None, description="Maximum apparent temperature in °C"),
    apparent_temperature_min: float = Query(None, description="Minimum apparent temperature in °C"),
    # Wind parameters
    wind_speed_10m_max: float = Query(None, description="Maximum wind speed at 10m"),
    wind_gusts_10m_max: float = Query(None, description="Maximum wind gusts at 10m"),
    wind_direction_10m_dominant: float = Query(None, description="Dominant wind direction at 10m"),
    wind_gusts_10m_mean: float = Query(None, description="Mean wind gusts at 10m"),
    wind_speed_10m_mean: float = Query(None, description="Mean wind speed at 10m"),
    wind_gusts_10m_min: float = Query(None, description="Minimum wind gusts at 10m"),
    wind_speed_10m_min: float = Query(None, description="Minimum wind speed at 10m"),
    winddirection_10m_dominant: float = Query(None, description="Dominant wind direction at 10m"),
    # Radiation parameters
    shortwave_radiation_sum: float = Query(None, description="Sum of shortwave radiation"),
    et0_fao_evapotranspiration: float = Query(None, description="FAO reference evapotranspiration"),
    et0_fao_evapotranspiration_sum: float = Query(None, description="Sum of FAO reference evapotranspiration"),
    # Sun parameters
    sunrise: int = Query(None, description="Sunrise time (Unix timestamp)"),
    sunset: int = Query(None, description="Sunset time (Unix timestamp)"),
    daylight_duration: float = Query(None, description="Daylight duration in seconds"),
    sunshine_duration: float = Query(None, description="Sunshine duration in seconds"),
    # Precipitation parameters
    rain_sum: float = Query(None, description="Rain sum in mm"),
    precipitation_sum: float = Query(None, description="Precipitation sum in mm"),
    snowfall_sum: float = Query(None, description="Snowfall sum in cm"),
    precipitation_hours: float = Query(None, description="Hours with precipitation"),
    snowfall_water_equivalent_sum: float = Query(None, description="Snowfall water equivalent sum"),
    # Cloud parameters
    cloud_cover_mean: float = Query(None, description="Mean cloud cover in %"),
    cloud_cover_max: float = Query(None, description="Maximum cloud cover in %"),
    cloud_cover_min: float = Query(None, description="Minimum cloud cover in %"),
    # Humidity parameters
    relative_humidity_2m_mean: float = Query(None, description="Mean relative humidity at 2m in %"),
    relative_humidity_2m_max: float = Query(None, description="Maximum relative humidity at 2m in %"),
    relative_humidity_2m_min: float = Query(None, description="Minimum relative humidity at 2m in %"),
    # Dew point parameters
    dew_point_2m_mean: float = Query(None, description="Mean dew point at 2m in °C"),
    dew_point_2m_max: float = Query(None, description="Maximum dew point at 2m in °C"),
    dew_point_2m_min: float = Query(None, description="Minimum dew point at 2m in °C"),
    # Pressure parameters
    pressure_msl_mean: float = Query(None, description="Mean sea level pressure in hPa"),
    pressure_msl_max: float = Query(None, description="Maximum sea level pressure in hPa"),
    pressure_msl_min: float = Query(None, description="Minimum sea level pressure in hPa"),
    surface_pressure_mean: float = Query(None, description="Mean surface pressure in hPa"),
    surface_pressure_max: float = Query(None, description="Maximum surface pressure in hPa"),
    surface_pressure_min: float = Query(None, description="Minimum surface pressure in hPa"),
    # Temperature derived parameters
    wet_bulb_temperature_2m_mean: float = Query(None, description="Mean wet bulb temperature at 2m in °C"),
    wet_bulb_temperature_2m_max: float = Query(None, description="Maximum wet bulb temperature at 2m in °C"),
    wet_bulb_temperature_2m_min: float = Query(None, description="Minimum wet bulb temperature at 2m in °C"),
    vapour_pressure_deficit_max: float = Query(None, description="Maximum vapour pressure deficit"),
    # Soil parameters
    soil_moisture_0_to_100cm_mean: float = Query(None, description="Mean soil moisture 0-100cm"),
    soil_moisture_0_to_7cm_mean: float = Query(None, description="Mean soil moisture 0-7cm"),
    soil_moisture_28_to_100cm_mean: float = Query(None, description="Mean soil moisture 28-100cm"),
    soil_moisture_7_to_28cm_mean: float = Query(None, description="Mean soil moisture 7-28cm"),
    soil_temperature_0_to_100cm_mean: float = Query(None, description="Mean soil temperature 0-100cm in °C"),
    soil_temperature_0_to_7cm_mean: float = Query(None, description="Mean soil temperature 0-7cm in °C"),
    soil_temperature_7_to_28cm_mean: float = Query(None, description="Mean soil temperature 7-28cm in °C"),
    soil_temperature_28_to_100cm_mean: float = Query(None, description="Mean soil temperature 28-100cm in °C")
) -> Dict[str, Any]:
    
    """
    Predict if it will rain in exactly 7 days from the given date.
    
    - Provide a date in YYYY-MM-DD format
    - Optionally provide weather parameters (defaults will be used if not provided)
    - Returns prediction whether it will rain 7 days from input date
    """
    # Validate input date
    input_date = validate_date_format(date)
    prediction_date = input_date + timedelta(days=7)
    
    # Check if rain model is loaded
    if rain_model is None:
        raise HTTPException(
            status_code=503,
            detail="Rain prediction model not available due to model loading error"
        )
    
    try:
        # Prepare weather data with provided parameters
        weather_params = {
            "date": date,
        }
        
        # Add non-None parameters

        if weather_code is not None:
            weather_params["weather_code"] = weather_code
        if temperature_mean is not None:
            weather_params["temperature_2m_mean"] = temperature_mean
        if temperature_max is not None:
            weather_params["temperature_2m_max"] = temperature_max
        if temperature_min is not None:
            weather_params["temperature_2m_min"] = temperature_min
        if rain_sum is not None:
            weather_params["rain_sum"] = rain_sum
        if precipitation_sum is not None:
            weather_params["precipitation_sum"] = precipitation_sum
            
        # Handle all remaining parameters
        if apparent_temperature_mean is not None:
            weather_params["apparent_temperature_mean"] = apparent_temperature_mean
        if apparent_temperature_max is not None:
            weather_params["apparent_temperature_max"] = apparent_temperature_max
        if apparent_temperature_min is not None:
            weather_params["apparent_temperature_min"] = apparent_temperature_min
        if wind_speed_10m_max is not None:
            weather_params["wind_speed_10m_max"] = wind_speed_10m_max
        if wind_gusts_10m_max is not None:
            weather_params["wind_gusts_10m_max"] = wind_gusts_10m_max
        if wind_direction_10m_dominant is not None:
            weather_params["wind_direction_10m_dominant"] = wind_direction_10m_dominant
        if shortwave_radiation_sum is not None:
            weather_params["shortwave_radiation_sum"] = shortwave_radiation_sum
        if et0_fao_evapotranspiration is not None:
            weather_params["et0_fao_evapotranspiration"] = et0_fao_evapotranspiration
        if sunrise is not None:
            weather_params["sunrise"] = sunrise
        if sunset is not None:
            weather_params["sunset"] = sunset
        if daylight_duration is not None:
            weather_params["daylight_duration"] = daylight_duration
        if sunshine_duration is not None:
            weather_params["sunshine_duration"] = sunshine_duration
        if snowfall_sum is not None:
            weather_params["snowfall_sum"] = snowfall_sum
        if precipitation_hours is not None:
            weather_params["precipitation_hours"] = precipitation_hours
        if cloud_cover_mean is not None:
            weather_params["cloud_cover_mean"] = cloud_cover_mean
        if cloud_cover_max is not None:
            weather_params["cloud_cover_max"] = cloud_cover_max
        if cloud_cover_min is not None:
            weather_params["cloud_cover_min"] = cloud_cover_min
        if dew_point_2m_mean is not None:
            weather_params["dew_point_2m_mean"] = dew_point_2m_mean
        if dew_point_2m_max is not None:
            weather_params["dew_point_2m_max"] = dew_point_2m_max
        if dew_point_2m_min is not None:
            weather_params["dew_point_2m_min"] = dew_point_2m_min
        if et0_fao_evapotranspiration_sum is not None:
            weather_params["et0_fao_evapotranspiration_sum"] = et0_fao_evapotranspiration_sum
        if relative_humidity_2m_mean is not None:
            weather_params["relative_humidity_2m_mean"] = relative_humidity_2m_mean
        if relative_humidity_2m_max is not None:
            weather_params["relative_humidity_2m_max"] = relative_humidity_2m_max
        if relative_humidity_2m_min is not None:
            weather_params["relative_humidity_2m_min"] = relative_humidity_2m_min
        if snowfall_water_equivalent_sum is not None:
            weather_params["snowfall_water_equivalent_sum"] = snowfall_water_equivalent_sum
        if pressure_msl_mean is not None:
            weather_params["pressure_msl_mean"] = pressure_msl_mean
        if pressure_msl_max is not None:
            weather_params["pressure_msl_max"] = pressure_msl_max
        if pressure_msl_min is not None:
            weather_params["pressure_msl_min"] = pressure_msl_min
        if surface_pressure_mean is not None:
            weather_params["surface_pressure_mean"] = surface_pressure_mean
        if surface_pressure_max is not None:
            weather_params["surface_pressure_max"] = surface_pressure_max
        if surface_pressure_min is not None:
            weather_params["surface_pressure_min"] = surface_pressure_min
        if winddirection_10m_dominant is not None:
            weather_params["winddirection_10m_dominant"] = winddirection_10m_dominant
        if wind_gusts_10m_mean is not None:
            weather_params["wind_gusts_10m_mean"] = wind_gusts_10m_mean
        if wind_speed_10m_mean is not None:
            weather_params["wind_speed_10m_mean"] = wind_speed_10m_mean
        if wind_gusts_10m_min is not None:
            weather_params["wind_gusts_10m_min"] = wind_gusts_10m_min
        if wind_speed_10m_min is not None:
            weather_params["wind_speed_10m_min"] = wind_speed_10m_min
        if wet_bulb_temperature_2m_mean is not None:
            weather_params["wet_bulb_temperature_2m_mean"] = wet_bulb_temperature_2m_mean
        if wet_bulb_temperature_2m_max is not None:
            weather_params["wet_bulb_temperature_2m_max"] = wet_bulb_temperature_2m_max
        if wet_bulb_temperature_2m_min is not None:
            weather_params["wet_bulb_temperature_2m_min"] = wet_bulb_temperature_2m_min
        if vapour_pressure_deficit_max is not None:
            weather_params["vapour_pressure_deficit_max"] = vapour_pressure_deficit_max
        if soil_moisture_0_to_100cm_mean is not None:
            weather_params["soil_moisture_0_to_100cm_mean"] = soil_moisture_0_to_100cm_mean
        if soil_moisture_0_to_7cm_mean is not None:
            weather_params["soil_moisture_0_to_7cm_mean"] = soil_moisture_0_to_7cm_mean
        if soil_moisture_28_to_100cm_mean is not None:
            weather_params["soil_moisture_28_to_100cm_mean"] = soil_moisture_28_to_100cm_mean
        if soil_moisture_7_to_28cm_mean is not None:
            weather_params["soil_moisture_7_to_28cm_mean"] = soil_moisture_7_to_28cm_mean
        if soil_temperature_0_to_100cm_mean is not None:
            weather_params["soil_temperature_0_to_100cm_mean"] = soil_temperature_0_to_100cm_mean
        if soil_temperature_0_to_7cm_mean is not None:
            weather_params["soil_temperature_0_to_7cm_mean"] = soil_temperature_0_to_7cm_mean
        if soil_temperature_7_to_28cm_mean is not None:
            weather_params["soil_temperature_7_to_28cm_mean"] = soil_temperature_7_to_28cm_mean
        if soil_temperature_28_to_100cm_mean is not None:
            weather_params["soil_temperature_28_to_100cm_mean"] = soil_temperature_28_to_100cm_mean
        
        # Format weather data with provided parameters and defaults for the rest
        weather_data = format_weather_data(**weather_params)
        
        # Handle categorical columns that might cause issues
        categorical_columns = weather_data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_columns:
            if col != 'date':  # Keep date as is
                weather_data[col] = pd.to_numeric(weather_data[col], errors='coerce')
        
        # Make prediction with proper error handling
        try:
            # Make prediction with the formatted data
            rain_prediction = rain_model.predict(weather_data, voting_method='majority')
            will_rain = bool(rain_prediction[0])
        except Exception as model_error:
            print(f"Error in model prediction: {model_error}")
            print(traceback.format_exc())
            
            # If there's an error with categorical columns, try dropping them
            if "pandas dtypes must be int, float or bool" in str(model_error):
                # Drop any non-numeric columns
                numeric_data = weather_data.select_dtypes(include=['number'])
                rain_prediction = rain_model.predict(numeric_data, voting_method='majority')
                will_rain = bool(rain_prediction[0])
            else:
                raise model_error
        
        # Return formatted response
        return {
            "input_date": input_date.strftime("%Y-%m-%d"),
            "prediction": {
                "date": prediction_date.strftime("%Y-%m-%d"),
                "will_rain": will_rain
            },
            "input_parameters": {k: v for k, v in weather_params.items() if k != 'date'}
        }
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error making rain prediction: {str(e)}"
        )
@app.get("/predict/precipitation/fall/")
async def predict_precipitation_fall(
    date: str = Query(None, description="Date in YYYY-MM-DD format (REQUIRED)"),
    # Weather code
    weather_code: float = Query(None, description="Weather code"),
    # Temperature parameters
    temperature_mean: float = Query(None, description="Mean temperature in °C", alias="temperature_2m_mean"),
    temperature_max: float = Query(None, description="Maximum temperature in °C", alias="temperature_2m_max"),
    temperature_min: float = Query(None, description="Minimum temperature in °C", alias="temperature_2m_min"),
    # Apparent temperature parameters
    apparent_temperature_mean: float = Query(None, description="Mean apparent temperature in °C"),
    apparent_temperature_max: float = Query(None, description="Maximum apparent temperature in °C"),
    apparent_temperature_min: float = Query(None, description="Minimum apparent temperature in °C"),
    # Wind parameters
    wind_speed_10m_max: float = Query(None, description="Maximum wind speed at 10m"),
    wind_gusts_10m_max: float = Query(None, description="Maximum wind gusts at 10m"),
    wind_direction_10m_dominant: float = Query(None, description="Dominant wind direction at 10m"),
    wind_gusts_10m_mean: float = Query(None, description="Mean wind gusts at 10m"),
    wind_speed_10m_mean: float = Query(None, description="Mean wind speed at 10m"),
    wind_gusts_10m_min: float = Query(None, description="Minimum wind gusts at 10m"),
    wind_speed_10m_min: float = Query(None, description="Minimum wind speed at 10m"),
    winddirection_10m_dominant: float = Query(None, description="Dominant wind direction at 10m"),
    # Radiation parameters
    shortwave_radiation_sum: float = Query(None, description="Sum of shortwave radiation"),
    et0_fao_evapotranspiration: float = Query(None, description="FAO reference evapotranspiration"),
    et0_fao_evapotranspiration_sum: float = Query(None, description="Sum of FAO reference evapotranspiration"),
    # Sun parameters
    sunrise: int = Query(None, description="Sunrise time (Unix timestamp)"),
    sunset: int = Query(None, description="Sunset time (Unix timestamp)"),
    daylight_duration: float = Query(None, description="Daylight duration in seconds"),
    sunshine_duration: float = Query(None, description="Sunshine duration in seconds"),
    # Precipitation parameters
    rain_sum: float = Query(None, description="Rain sum in mm"),
    precipitation_sum: float = Query(None, description="Precipitation sum in mm"),
    snowfall_sum: float = Query(None, description="Snowfall sum in cm"),
    precipitation_hours: float = Query(None, description="Hours with precipitation"),
    snowfall_water_equivalent_sum: float = Query(None, description="Snowfall water equivalent sum"),
    # Cloud parameters
    cloud_cover_mean: float = Query(None, description="Mean cloud cover in %"),
    cloud_cover_max: float = Query(None, description="Maximum cloud cover in %"),
    cloud_cover_min: float = Query(None, description="Minimum cloud cover in %"),
    # Humidity parameters
    relative_humidity_2m_mean: float = Query(None, description="Mean relative humidity at 2m in %"),
    relative_humidity_2m_max: float = Query(None, description="Maximum relative humidity at 2m in %"),
    relative_humidity_2m_min: float = Query(None, description="Minimum relative humidity at 2m in %"),
    # Dew point parameters
    dew_point_2m_mean: float = Query(None, description="Mean dew point at 2m in °C"),
    dew_point_2m_max: float = Query(None, description="Maximum dew point at 2m in °C"),
    dew_point_2m_min: float = Query(None, description="Minimum dew point at 2m in °C"),
    # Pressure parameters
    pressure_msl_mean: float = Query(None, description="Mean sea level pressure in hPa"),
    pressure_msl_max: float = Query(None, description="Maximum sea level pressure in hPa"),
    pressure_msl_min: float = Query(None, description="Minimum sea level pressure in hPa"),
    surface_pressure_mean: float = Query(None, description="Mean surface pressure in hPa"),
    surface_pressure_max: float = Query(None, description="Maximum surface pressure in hPa"),
    surface_pressure_min: float = Query(None, description="Minimum surface pressure in hPa"),
    # Temperature derived parameters
    wet_bulb_temperature_2m_mean: float = Query(None, description="Mean wet bulb temperature at 2m in °C"),
    wet_bulb_temperature_2m_max: float = Query(None, description="Maximum wet bulb temperature at 2m in °C"),
    wet_bulb_temperature_2m_min: float = Query(None, description="Minimum wet bulb temperature at 2m in °C"),
    vapour_pressure_deficit_max: float = Query(None, description="Maximum vapour pressure deficit"),
    # Soil parameters
    soil_moisture_0_to_100cm_mean: float = Query(None, description="Mean soil moisture 0-100cm"),
    soil_moisture_0_to_7cm_mean: float = Query(None, description="Mean soil moisture 0-7cm"),
    soil_moisture_28_to_100cm_mean: float = Query(None, description="Mean soil moisture 28-100cm"),
    soil_moisture_7_to_28cm_mean: float = Query(None, description="Mean soil moisture 7-28cm"),
    soil_temperature_0_to_100cm_mean: float = Query(None, description="Mean soil temperature 0-100cm in °C"),
    soil_temperature_0_to_7cm_mean: float = Query(None, description="Mean soil temperature 0-7cm in °C"),
    soil_temperature_7_to_28cm_mean: float = Query(None, description="Mean soil temperature 7-28cm in °C"),
    soil_temperature_28_to_100cm_mean: float = Query(None, description="Mean soil temperature 28-100cm in °C")
) -> Dict[str, Any]:
    """
    Predict cumulated sum of precipitation (in mm) within the next 3 days.
    
    - You must provide a date in YYYY-MM-DD format
    - Optionally provide weather parameters (defaults will be used if not provided)
    - Returns prediction of precipitation amount for the 3 days following input date
    
    Example usage:
    - Basic: /predict/precipitation/fall/?date=2023-09-25
    - With parameters: /predict/precipitation/fall/?date=2023-09-25&temperature_mean=22.5&rain_sum=0.5
    """
    # Check if date is provided
    if date is None:
        return {
            "message": "Please provide a date to make a precipitation prediction",
            "required_parameters": {
                "date": "Date in YYYY-MM-DD format (REQUIRED)"
            },
            "optional_parameters": {
                "temperature_mean": "Mean temperature in °C",
                "temperature_max": "Maximum temperature in °C",
                "temperature_min": "Minimum temperature in °C",
                "wind_speed_10m_mean": "Mean wind speed at 10m",
                "relative_humidity_2m_mean": "Mean relative humidity at 2m in %",
                # Add more key parameters here
            },
            "example_request": "/predict/precipitation/fall/?date=2023-09-25&temperature_mean=22.5&relative_humidity_2m_mean=65",
            "note": "All weather parameters are optional. Default values will be used for any parameters not provided."
        }
    
    # Validate input date
    try:
        input_date = validate_date_format(date)
        start_date = input_date + timedelta(days=1)
        end_date = input_date + timedelta(days=3)
    except HTTPException as e:
        return {
            "error": e.detail,
            "message": "Please provide a valid date in YYYY-MM-DD format",
            "example": "2023-09-25"
        }
    
    # Check if precipitation model is loaded
    if precipitation_model is None:
        raise HTTPException(
            status_code=503,
            detail="Precipitation prediction model not available. Please try again later."
        )
    
    try:
        # Prepare weather data with provided parameters
        weather_params = {
            "date": date,
        }
        
        # Add non-None parameters
        if weather_code is not None:
            weather_params["weather_code"] = weather_code
        if temperature_mean is not None:
            weather_params["temperature_2m_mean"] = temperature_mean
        if temperature_max is not None:
            weather_params["temperature_2m_max"] = temperature_max
        if temperature_min is not None:
            weather_params["temperature_2m_min"] = temperature_min
        if rain_sum is not None:
            weather_params["rain_sum"] = rain_sum
        if precipitation_sum is not None:
            weather_params["precipitation_sum"] = precipitation_sum
            
        # Handle all remaining parameters
        if apparent_temperature_mean is not None:
            weather_params["apparent_temperature_mean"] = apparent_temperature_mean
        if apparent_temperature_max is not None:
            weather_params["apparent_temperature_max"] = apparent_temperature_max
        if apparent_temperature_min is not None:
            weather_params["apparent_temperature_min"] = apparent_temperature_min
        if wind_speed_10m_max is not None:
            weather_params["wind_speed_10m_max"] = wind_speed_10m_max
        if wind_gusts_10m_max is not None:
            weather_params["wind_gusts_10m_max"] = wind_gusts_10m_max
        if wind_direction_10m_dominant is not None:
            weather_params["wind_direction_10m_dominant"] = wind_direction_10m_dominant
        if shortwave_radiation_sum is not None:
            weather_params["shortwave_radiation_sum"] = shortwave_radiation_sum
        if et0_fao_evapotranspiration is not None:
            weather_params["et0_fao_evapotranspiration"] = et0_fao_evapotranspiration
        if sunrise is not None:
            weather_params["sunrise"] = sunrise
        if sunset is not None:
            weather_params["sunset"] = sunset
        if daylight_duration is not None:
            weather_params["daylight_duration"] = daylight_duration
        if sunshine_duration is not None:
            weather_params["sunshine_duration"] = sunshine_duration
        if snowfall_sum is not None:
            weather_params["snowfall_sum"] = snowfall_sum
        if precipitation_hours is not None:
            weather_params["precipitation_hours"] = precipitation_hours
        if cloud_cover_mean is not None:
            weather_params["cloud_cover_mean"] = cloud_cover_mean
        if cloud_cover_max is not None:
            weather_params["cloud_cover_max"] = cloud_cover_max
        if cloud_cover_min is not None:
            weather_params["cloud_cover_min"] = cloud_cover_min
        if dew_point_2m_mean is not None:
            weather_params["dew_point_2m_mean"] = dew_point_2m_mean
        if dew_point_2m_max is not None:
            weather_params["dew_point_2m_max"] = dew_point_2m_max
        if dew_point_2m_min is not None:
            weather_params["dew_point_2m_min"] = dew_point_2m_min
        if et0_fao_evapotranspiration_sum is not None:
            weather_params["et0_fao_evapotranspiration_sum"] = et0_fao_evapotranspiration_sum
        if relative_humidity_2m_mean is not None:
            weather_params["relative_humidity_2m_mean"] = relative_humidity_2m_mean
        if relative_humidity_2m_max is not None:
            weather_params["relative_humidity_2m_max"] = relative_humidity_2m_max
        if relative_humidity_2m_min is not None:
            weather_params["relative_humidity_2m_min"] = relative_humidity_2m_min
        if snowfall_water_equivalent_sum is not None:
            weather_params["snowfall_water_equivalent_sum"] = snowfall_water_equivalent_sum
        if pressure_msl_mean is not None:
            weather_params["pressure_msl_mean"] = pressure_msl_mean
        if pressure_msl_max is not None:
            weather_params["pressure_msl_max"] = pressure_msl_max
        if pressure_msl_min is not None:
            weather_params["pressure_msl_min"] = pressure_msl_min
        if surface_pressure_mean is not None:
            weather_params["surface_pressure_mean"] = surface_pressure_mean
        if surface_pressure_max is not None:
            weather_params["surface_pressure_max"] = surface_pressure_max
        if surface_pressure_min is not None:
            weather_params["surface_pressure_min"] = surface_pressure_min
        if winddirection_10m_dominant is not None:
            weather_params["winddirection_10m_dominant"] = winddirection_10m_dominant
        if wind_gusts_10m_mean is not None:
            weather_params["wind_gusts_10m_mean"] = wind_gusts_10m_mean
        if wind_speed_10m_mean is not None:
            weather_params["wind_speed_10m_mean"] = wind_speed_10m_mean
        if wind_gusts_10m_min is not None:
            weather_params["wind_gusts_10m_min"] = wind_gusts_10m_min
        if wind_speed_10m_min is not None:
            weather_params["wind_speed_10m_min"] = wind_speed_10m_min
        if wet_bulb_temperature_2m_mean is not None:
            weather_params["wet_bulb_temperature_2m_mean"] = wet_bulb_temperature_2m_mean
        if wet_bulb_temperature_2m_max is not None:
            weather_params["wet_bulb_temperature_2m_max"] = wet_bulb_temperature_2m_max
        if wet_bulb_temperature_2m_min is not None:
            weather_params["wet_bulb_temperature_2m_min"] = wet_bulb_temperature_2m_min
        if vapour_pressure_deficit_max is not None:
            weather_params["vapour_pressure_deficit_max"] = vapour_pressure_deficit_max
        if soil_moisture_0_to_100cm_mean is not None:
            weather_params["soil_moisture_0_to_100cm_mean"] = soil_moisture_0_to_100cm_mean
        if soil_moisture_0_to_7cm_mean is not None:
            weather_params["soil_moisture_0_to_7cm_mean"] = soil_moisture_0_to_7cm_mean
        if soil_moisture_28_to_100cm_mean is not None:
            weather_params["soil_moisture_28_to_100cm_mean"] = soil_moisture_28_to_100cm_mean
        if soil_moisture_7_to_28cm_mean is not None:
            weather_params["soil_moisture_7_to_28cm_mean"] = soil_moisture_7_to_28cm_mean
        if soil_temperature_0_to_100cm_mean is not None:
            weather_params["soil_temperature_0_to_100cm_mean"] = soil_temperature_0_to_100cm_mean
        if soil_temperature_0_to_7cm_mean is not None:
            weather_params["soil_temperature_0_to_7cm_mean"] = soil_temperature_0_to_7cm_mean
        if soil_temperature_7_to_28cm_mean is not None:
            weather_params["soil_temperature_7_to_28cm_mean"] = soil_temperature_7_to_28cm_mean
        if soil_temperature_28_to_100cm_mean is not None:
            weather_params["soil_temperature_28_to_100cm_mean"] = soil_temperature_28_to_100cm_mean
            
        # Format weather data with provided parameters and defaults for the rest
        weather_data = format_weather_data(**weather_params)
        
        # Handle categorical columns that might cause issues
        categorical_columns = weather_data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_columns:
            if col != 'date':  # Keep date as is
                weather_data[col] = pd.to_numeric(weather_data[col], errors='coerce')
        
        # Make prediction with proper error handling
        try:
            # Make prediction with the formatted data
            precipitation = precipitation_model.predict_ensemble(weather_data, method='mean')
        except Exception as model_error:
            print(f"Error in precipitation model prediction: {model_error}")
            print(traceback.format_exc())
            
            # If there's an error with categorical columns, try dropping them
            if "pandas dtypes must be int, float or bool" in str(model_error):
                # Drop any non-numeric columns
                numeric_data = weather_data.select_dtypes(include=['number'])
                precipitation = precipitation_model.predict_ensemble(numeric_data, method='mean')
            else:
                raise model_error
        
        # Get first prediction result and ensure it's positive
        precipitation_amount = max(0.0, float(precipitation[0]))
        
        # Return formatted response
        return {
            "input_date": input_date.strftime("%Y-%m-%d"),
            "prediction": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "precipitation_fall": f"{precipitation_amount:.1f}"
            },
            "input_parameters": {k: v for k, v in weather_params.items() if k != 'date'}
        }
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error making precipitation prediction: {str(e)}"
        )




###########################################################################################################################################################################

# Start the application if run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)