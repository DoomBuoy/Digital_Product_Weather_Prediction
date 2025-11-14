import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Check if the API is live
st.markdown("🔍 Checking API status...")
try:
    response = requests.get("https://weather-prediction-api-isp4.onrender.com/health/", timeout=10)
    if response.status_code == 200:
        st.success("✅ API is awake and ready for prediction")
    else:
        st.error("⏳ Please wait for 2 min, the app is waking up the prediction API. Refresh the page after a couple of minutes.")
        st.stop()
except requests.exceptions.RequestException:
    st.error("❌ Unable to connect to the prediction API. Please check your internet connection and try again later.")
    st.stop()

# Setup for weather data fetching
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

API_BASE_URL = "https://weather-prediction-api-isp4.onrender.com"

# List of features
features = [
    "date", "weather_code", "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration", "sunrise", "sunset",
    "daylight_duration", "sunshine_duration", "rain_sum", "precipitation_sum",
    "snowfall_sum", "precipitation_hours", "cloud_cover_mean", "cloud_cover_max",
    "cloud_cover_min", "dew_point_2m_mean", "dew_point_2m_max", "dew_point_2m_min",
    "et0_fao_evapotranspiration_sum", "relative_humidity_2m_mean", "relative_humidity_2m_max",
    "relative_humidity_2m_min", "snowfall_water_equivalent_sum", "pressure_msl_mean",
    "pressure_msl_max", "pressure_msl_min", "surface_pressure_mean", "surface_pressure_max",
    "surface_pressure_min", "winddirection_10m_dominant", "wind_gusts_10m_mean",
    "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min",
    "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min",
    "vapour_pressure_deficit_max", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean",
    "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean",
    "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean",
    "soil_temperature_7_to_28cm_mean", "soil_temperature_28_to_100cm_mean"
]

# Default values for features (from main.py)
defaults = {
    "weather_code": 51.0,
    "temperature_2m_mean": 17.627083,
    "temperature_2m_max": 21.3375,
    "temperature_2m_min": 13.9375,
    "apparent_temperature_mean": 16.776638,
    "apparent_temperature_max": 21.033524,
    "apparent_temperature_min": 12.8664055,
    "wind_speed_10m_max": 18.345877,
    "wind_gusts_10m_max": 30.0,  # approximate
    "wind_direction_10m_dominant": 180.0,
    "shortwave_radiation_sum": 15.0,
    "et0_fao_evapotranspiration": 3.0,
    "sunrise": 360,  # minutes since midnight
    "sunset": 1080,
    "daylight_duration": 12.0,
    "sunshine_duration": 8.0,
    "rain_sum": 0.0,
    "precipitation_sum": 0.0,
    "snowfall_sum": 0.0,
    "precipitation_hours": 0.0,
    "cloud_cover_mean": 50.0,
    "cloud_cover_max": 80.0,
    "cloud_cover_min": 20.0,
    "dew_point_2m_mean": 12.0,
    "dew_point_2m_max": 15.0,
    "dew_point_2m_min": 10.0,
    "et0_fao_evapotranspiration_sum": 3.0,
    "relative_humidity_2m_mean": 70.0,
    "relative_humidity_2m_max": 85.0,
    "relative_humidity_2m_min": 55.0,
    "snowfall_water_equivalent_sum": 0.0,
    "pressure_msl_mean": 1013.0,
    "pressure_msl_max": 1020.0,
    "pressure_msl_min": 1005.0,
    "surface_pressure_mean": 1010.0,
    "surface_pressure_max": 1017.0,
    "surface_pressure_min": 1002.0,
    "winddirection_10m_dominant": 180.0,
    "wind_gusts_10m_mean": 15.0,
    "wind_speed_10m_mean": 10.0,
    "wind_gusts_10m_min": 5.0,
    "wind_speed_10m_min": 2.0,
    "wet_bulb_temperature_2m_mean": 15.0,
    "wet_bulb_temperature_2m_max": 18.0,
    "wet_bulb_temperature_2m_min": 12.0,
    "vapour_pressure_deficit_max": 1.0,
    "soil_moisture_0_to_100cm_mean": 0.2,
    "soil_moisture_0_to_7cm_mean": 0.25,
    "soil_moisture_28_to_100cm_mean": 0.15,
    "soil_moisture_7_to_28cm_mean": 0.2,
    "soil_temperature_0_to_100cm_mean": 18.0,
    "soil_temperature_0_to_7cm_mean": 20.0,
    "soil_temperature_7_to_28cm_mean": 17.0,
    "soil_temperature_28_to_100cm_mean": 16.0
}

def fetch_weather_data(date_str):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": -33.8678,
        "longitude": 151.2073,
        "start_date": date_str,
        "end_date": date_str,
        "daily": [
            "weather_code", "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min",
            "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration", "sunrise", "sunset",
            "daylight_duration", "sunshine_duration", "rain_sum", "precipitation_sum",
            "snowfall_sum", "precipitation_hours", "cloud_cover_mean", "cloud_cover_max",
            "cloud_cover_min", "dew_point_2m_mean", "dew_point_2m_max", "dew_point_2m_min",
            "et0_fao_evapotranspiration_sum", "relative_humidity_2m_mean", "relative_humidity_2m_max",
            "relative_humidity_2m_min", "snowfall_water_equivalent_sum", "pressure_msl_mean",
            "pressure_msl_max", "pressure_msl_min", "surface_pressure_mean", "surface_pressure_max",
            "surface_pressure_min", "winddirection_10m_dominant", "wind_gusts_10m_mean",
            "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min",
            "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_min",
            "vapour_pressure_deficit_max", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean",
            "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean",
            "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean",
            "soil_temperature_7_to_28cm_mean", "soil_temperature_28_to_100cm_mean"
        ],
        "timezone": "auto",
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()
        data = {}
        for i, var in enumerate(params["daily"]):
            values = daily.Variables(i).ValuesAsNumpy()
            if hasattr(values, '__len__') and len(values) > 0:
                value = values[0]
            else:
                value = values  # scalar
            if var in ["sunrise", "sunset"]:
                try:
                    data[var] = int(value)
                except ValueError:
                    h, m = map(int, value.split(":"))
                    data[var] = h * 60 + m
            else:
                data[var] = float(value)
        return data
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def predict_rain(date_str, params):
    url = f"{API_BASE_URL}/predict/rain/"
    params["date"] = date_str
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None

def predict_precipitation(date_str, params):
    url = f"{API_BASE_URL}/predict/precipitation/fall/"
    params["date"] = date_str
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None

st.title("🌤️ Weather Prediction App for Sydney")

st.markdown("Predict rain and precipitation using machine learning models based on historical weather data.")

st.divider()

st.header("1. 🌧️ Rain Prediction (Will it rain exactly 7 days from the input date?)")

rain_option = st.radio("Choose Data Source for Rain Prediction", ["Real-time Data", "Custom Input"], key="rain", horizontal=True)

if rain_option == "Real-time Data":
    st.subheader("Real-time Prediction")
    date = st.date_input("Select Base Date (data fetched for this date)", (datetime.now() - timedelta(days=1)).date(), key="rain_date")
    if st.button("🔮 Predict Rain with Real-time Data", key="rain_predict"):
        with st.spinner("Fetching data and predicting..."):
            data = fetch_weather_data(date.strftime("%Y-%m-%d"))
        if data:
            with st.spinner("Getting prediction..."):
                result = predict_rain(date.strftime("%Y-%m-%d"), data)
            if result:
                pred = result['prediction']
                if pred['will_rain']:
                    st.markdown(f"<h2 style='color: blue; text-align: center;'>🌧️ Will it rain exactly 7 days from the input date (on {pred['date']})? <strong style='color: green;'>YES</strong></h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color: blue; text-align: center;'>☀️ Will it rain exactly 7 days from the input date (on {pred['date']})? <strong style='color: red;'>NO</strong></h2>", unsafe_allow_html=True)
                with st.expander("📄 View Raw JSON Response"):
                    st.json(result)
else:
    st.subheader("Custom Input Prediction")
    with st.form("rain_custom_form"):
        st.write("Enter custom values for weather features:")
        cols = st.columns(3)
        custom_params = {}
        for i, feature in enumerate(features[1:]):  # skip date
            with cols[i % 3]:
                custom_params[feature] = st.number_input(f"{feature}", value=defaults.get(feature, 0.0), key=f"rain_{feature}")
        date = st.date_input("Select Date", datetime.now().date(), key="rain_custom_date")
        submitted = st.form_submit_button("🔮 Predict Rain with Custom Data")
        if submitted:
            with st.spinner("Getting prediction..."):
                result = predict_rain(date.strftime("%Y-%m-%d"), custom_params)
            if result:
                pred = result['prediction']
                if pred['will_rain']:
                    st.markdown(f"<h2 style='color: blue; text-align: center;'>🌧️ Will it rain exactly 7 days from the input date (on {pred['date']})? <strong style='color: green;'>YES</strong></h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color: blue; text-align: center;'>☀️ Will it rain exactly 7 days from the input date (on {pred['date']})? <strong style='color: red;'>NO</strong></h2>", unsafe_allow_html=True)
                with st.expander("📄 View Raw JSON Response"):
                    st.json(result)

st.divider()

st.header("2. 💧 Precipitation Prediction (Cumulated precipitation in mm for next 3 days)")

precip_option = st.radio("Choose Data Source for Precipitation Prediction", ["Real-time Data", "Custom Input"], key="precip", horizontal=True)

if precip_option == "Real-time Data":
    st.subheader("Real-time Prediction")
    date = st.date_input("Select Base Date (data fetched for this date)", (datetime.now() - timedelta(days=1)).date(), key="precip_date")
    if st.button("🔮 Predict Precipitation with Real-time Data", key="precip_predict"):
        with st.spinner("Fetching data and predicting..."):
            data = fetch_weather_data(date.strftime("%Y-%m-%d"))
        if data:
            with st.spinner("Getting prediction..."):
                result = predict_precipitation(date.strftime("%Y-%m-%d"), data)
            if result:
                pred = result['prediction']
                st.markdown(f"<h2 style='color: blue; text-align: center;'>💧 Predicted precipitation from {pred['start_date']} to {pred['end_date']}: <strong style='color: purple;'>{pred['precipitation_fall']} mm</strong></h2>", unsafe_allow_html=True)
                with st.expander("📄 View Raw JSON Response"):
                    st.json(result)
else:
    st.subheader("Custom Input Prediction")
    with st.form("precip_custom_form"):
        st.write("Enter custom values for weather features:")
        cols = st.columns(3)
        custom_params = {}
        for i, feature in enumerate(features[1:]):  # skip date
            with cols[i % 3]:
                custom_params[feature] = st.number_input(f"{feature}", value=defaults.get(feature, 0.0), key=f"precip_{feature}")
        date = st.date_input("Select Date", datetime.now().date(), key="precip_custom_date")
        submitted = st.form_submit_button("🔮 Predict Precipitation with Custom Data")
        if submitted:
            with st.spinner("Getting prediction..."):
                result = predict_precipitation(date.strftime("%Y-%m-%d"), custom_params)
            if result:
                pred = result['prediction']
                st.markdown(f"<h2 style='color: blue; text-align: center;'>💧 Predicted precipitation from {pred['start_date']} to {pred['end_date']}: <strong style='color: purple;'>{pred['precipitation_fall']} mm</strong></h2>", unsafe_allow_html=True)
                with st.expander("📄 View Raw JSON Response"):
                    st.json(result)
