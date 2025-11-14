from pathlib import Path
import os 
os.system('pip install poetry')
os.system('poetry update')
os.system('poetry install')
print("All dependencies are installed!")
from loguru import logger
from tqdm import tqdm
import typer
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from ml_as_api.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Create data directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": -33.8678,
        "longitude": 151.2073,
        "start_date": "2010-01-01",
        "end_date": "2024-12-31",
        "daily": ["weather_code", "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "sunrise", "sunset", "daylight_duration", "sunshine_duration", "rain_sum", "precipitation_sum", "snowfall_sum", "precipitation_hours", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min", "dew_point_2m_mean", "dew_point_2m_max", "dew_point_2m_min", "et0_fao_evapotranspiration_sum", "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min", "snowfall_water_equivalent_sum", "pressure_msl_mean", "pressure_msl_max", "pressure_msl_min", "surface_pressure_mean", "surface_pressure_max", "surface_pressure_min", "winddirection_10m_dominant", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min", "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean", "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_7_to_28cm_mean", "soil_temperature_28_to_100cm_mean"],
        "timezone": "auto",
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_weather_code = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_max = daily.Variables(2).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(3).ValuesAsNumpy()
    daily_apparent_temperature_mean = daily.Variables(4).ValuesAsNumpy()
    daily_apparent_temperature_max = daily.Variables(5).ValuesAsNumpy()
    daily_apparent_temperature_min = daily.Variables(6).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(7).ValuesAsNumpy()
    daily_wind_gusts_10m_max = daily.Variables(8).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(9).ValuesAsNumpy()
    daily_shortwave_radiation_sum = daily.Variables(10).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration = daily.Variables(11).ValuesAsNumpy()
    daily_sunrise = daily.Variables(12).ValuesInt64AsNumpy()
    daily_sunset = daily.Variables(13).ValuesInt64AsNumpy()
    daily_daylight_duration = daily.Variables(14).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(15).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(16).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(17).ValuesAsNumpy()
    daily_snowfall_sum = daily.Variables(18).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(19).ValuesAsNumpy()
    daily_cloud_cover_mean = daily.Variables(20).ValuesAsNumpy()
    daily_cloud_cover_max = daily.Variables(21).ValuesAsNumpy()
    daily_cloud_cover_min = daily.Variables(22).ValuesAsNumpy()
    daily_dew_point_2m_mean = daily.Variables(23).ValuesAsNumpy()
    daily_dew_point_2m_max = daily.Variables(24).ValuesAsNumpy()
    daily_dew_point_2m_min = daily.Variables(25).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration_sum = daily.Variables(26).ValuesAsNumpy()
    daily_relative_humidity_2m_mean = daily.Variables(27).ValuesAsNumpy()
    daily_relative_humidity_2m_max = daily.Variables(28).ValuesAsNumpy()
    daily_relative_humidity_2m_min = daily.Variables(29).ValuesAsNumpy()
    daily_snowfall_water_equivalent_sum = daily.Variables(30).ValuesAsNumpy()
    daily_pressure_msl_mean = daily.Variables(31).ValuesAsNumpy()
    daily_pressure_msl_max = daily.Variables(32).ValuesAsNumpy()
    daily_pressure_msl_min = daily.Variables(33).ValuesAsNumpy()
    daily_surface_pressure_mean = daily.Variables(34).ValuesAsNumpy()
    daily_surface_pressure_max = daily.Variables(35).ValuesAsNumpy()
    daily_surface_pressure_min = daily.Variables(36).ValuesAsNumpy()
    daily_winddirection_10m_dominant = daily.Variables(37).ValuesAsNumpy()
    daily_wind_gusts_10m_mean = daily.Variables(38).ValuesAsNumpy()
    daily_wind_speed_10m_mean = daily.Variables(39).ValuesAsNumpy()
    daily_wind_gusts_10m_min = daily.Variables(40).ValuesAsNumpy()
    daily_wind_speed_10m_min = daily.Variables(41).ValuesAsNumpy()
    daily_wet_bulb_temperature_2m_mean = daily.Variables(42).ValuesAsNumpy()
    daily_wet_bulb_temperature_2m_max = daily.Variables(43).ValuesAsNumpy()
    daily_wet_bulb_temperature_2m_min = daily.Variables(44).ValuesAsNumpy()
    daily_vapour_pressure_deficit_max = daily.Variables(45).ValuesAsNumpy()
    daily_soil_moisture_0_to_100cm_mean = daily.Variables(46).ValuesAsNumpy()
    daily_soil_moisture_0_to_7cm_mean = daily.Variables(47).ValuesAsNumpy()
    daily_soil_moisture_28_to_100cm_mean = daily.Variables(48).ValuesAsNumpy()
    daily_soil_moisture_7_to_28cm_mean = daily.Variables(49).ValuesAsNumpy()
    daily_soil_temperature_0_to_100cm_mean = daily.Variables(50).ValuesAsNumpy()
    daily_soil_temperature_0_to_7cm_mean = daily.Variables(51).ValuesAsNumpy()
    daily_soil_temperature_7_to_28cm_mean = daily.Variables(52).ValuesAsNumpy()
    daily_soil_temperature_28_to_100cm_mean = daily.Variables(53).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["weather_code"] = daily_weather_code
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
    daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
    daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
    daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
    daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
    daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration
    daily_data["sunrise"] = daily_sunrise
    daily_data["sunset"] = daily_sunset
    daily_data["daylight_duration"] = daily_daylight_duration
    daily_data["sunshine_duration"] = daily_sunshine_duration
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["snowfall_sum"] = daily_snowfall_sum
    daily_data["precipitation_hours"] = daily_precipitation_hours
    daily_data["cloud_cover_mean"] = daily_cloud_cover_mean
    daily_data["cloud_cover_max"] = daily_cloud_cover_max
    daily_data["cloud_cover_min"] = daily_cloud_cover_min
    daily_data["dew_point_2m_mean"] = daily_dew_point_2m_mean
    daily_data["dew_point_2m_max"] = daily_dew_point_2m_max
    daily_data["dew_point_2m_min"] = daily_dew_point_2m_min
    daily_data["et0_fao_evapotranspiration_sum"] = daily_et0_fao_evapotranspiration_sum
    daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
    daily_data["relative_humidity_2m_max"] = daily_relative_humidity_2m_max
    daily_data["relative_humidity_2m_min"] = daily_relative_humidity_2m_min
    daily_data["snowfall_water_equivalent_sum"] = daily_snowfall_water_equivalent_sum
    daily_data["pressure_msl_mean"] = daily_pressure_msl_mean
    daily_data["pressure_msl_max"] = daily_pressure_msl_max
    daily_data["pressure_msl_min"] = daily_pressure_msl_min
    daily_data["surface_pressure_mean"] = daily_surface_pressure_mean
    daily_data["surface_pressure_max"] = daily_surface_pressure_max
    daily_data["surface_pressure_min"] = daily_surface_pressure_min
    daily_data["winddirection_10m_dominant"] = daily_winddirection_10m_dominant
    daily_data["wind_gusts_10m_mean"] = daily_wind_gusts_10m_mean
    daily_data["wind_speed_10m_mean"] = daily_wind_speed_10m_mean
    daily_data["wind_gusts_10m_min"] = daily_wind_gusts_10m_min
    daily_data["wind_speed_10m_min"] = daily_wind_speed_10m_min
    daily_data["wet_bulb_temperature_2m_mean"] = daily_wet_bulb_temperature_2m_mean
    daily_data["wet_bulb_temperature_2m_max"] = daily_wet_bulb_temperature_2m_max
    daily_data["wet_bulb_temperature_2m_min"] = daily_wet_bulb_temperature_2m_min
    daily_data["vapour_pressure_deficit_max"] = daily_vapour_pressure_deficit_max
    daily_data["soil_moisture_0_to_100cm_mean"] = daily_soil_moisture_0_to_100cm_mean
    daily_data["soil_moisture_0_to_7cm_mean"] = daily_soil_moisture_0_to_7cm_mean
    daily_data["soil_moisture_28_to_100cm_mean"] = daily_soil_moisture_28_to_100cm_mean
    daily_data["soil_moisture_7_to_28cm_mean"] = daily_soil_moisture_7_to_28cm_mean
    daily_data["soil_temperature_0_to_100cm_mean"] = daily_soil_temperature_0_to_100cm_mean
    daily_data["soil_temperature_0_to_7cm_mean"] = daily_soil_temperature_0_to_7cm_mean
    daily_data["soil_temperature_7_to_28cm_mean"] = daily_soil_temperature_7_to_28cm_mean
    daily_data["soil_temperature_28_to_100cm_mean"] = daily_soil_temperature_28_to_100cm_mean

    daily_dataframe = pd.DataFrame(data = daily_data)
    
    
    # Save the weather data to CSV file in raw data directory
    output_file = RAW_DATA_DIR / "weather_data.csv"
    daily_dataframe.to_csv(output_file, index=False)
    logger.info(f"Weather data saved to: {output_file}")
    
    logger.success("Processing dataset complete.")
    print("\nDaily data\n", daily_dataframe)
    # -----------------------------------------


if __name__ == "__main__":
    app()
