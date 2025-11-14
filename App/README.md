# Weather Prediction App for Sydney

A Streamlit-based web application that predicts rain and precipitation for Sydney using machine learning models trained on historical weather data.

## Features

- **Rain Prediction**: Predicts whether it will rain exactly 7 days from a given date
- **Precipitation Prediction**: Predicts cumulative precipitation in mm for the next 3 days
- **Real-time Data**: Fetches current weather data from Open-Meteo API
- **Custom Input**: Allows manual input of weather parameters for predictions
- **API Integration**: Connects to a deployed prediction API for model inference

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DoomBuoy/Digital_Product_Weather_Prediction.git
   cd Digital_Product_Weather_Prediction/App
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

   Or using pip (if you prefer):
   ```bash
   pip install streamlit requests pandas openmeteo-requests requests-cache retry-requests
   ```

## Usage

1. Ensure the prediction API is running (hosted at https://weather-prediction-api-isp4.onrender.com)

2. Run the Streamlit app:
   ```bash
   streamlit run App_main.py
   ```

3. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

4. Choose between "Real-time Data" or "Custom Input" for predictions

5. For real-time predictions, select a base date and click predict

6. For custom input, fill in the weather parameters and select a date

## Dependencies

- streamlit >=1.51.0
- requests >=2.32.5
- pandas >=2.3.3
- openmeteo-requests >=1.7.4
- requests-cache >=1.2.1
- retry-requests >=2.0.0

## API Endpoints

The app communicates with the following API endpoints:
- `GET /health/` - Health check
- `GET /predict/rain/` - Rain prediction
- `GET /predict/precipitation/fall/` - Precipitation prediction

## Data Source

Weather data is fetched from the Open-Meteo Archive API for Sydney coordinates (latitude: -33.8678, longitude: 151.2073).

## Author

Agam Singh Saini - AgamSingh.Saini@student.uts.edu.au

## License

This project is part of a digital product development course at University of Technology Sydney.</content>
<parameter name="filePath">d:\MS_DSI\GithubProjects\Digital_Product_Weather_Prediction\App\README.md