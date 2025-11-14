# Weather Prediction API Project

## Overview
This project provides a robust, production-ready API for weather prediction using advanced machine learning models. It is designed for both technical and non-technical (HR) audiences, offering clear documentation, easy deployment, and reliable predictions based on historical weather data.

---

## For HR/Non-Technical Audience

**What does this project do?**
- Predicts weather outcomes (e.g., precipitation amount, rain/no rain) using historical weather data.
- Provides a simple API interface for integration with other systems or dashboards.
- Ensures reliability and consistency through robust data processing and model validation.

**Key Benefits:**
- Easy to use: Just send weather data to the API and get predictions.
- Well-documented: Clear instructions and endpoint descriptions.
- Ready for deployment: Can be integrated into business processes or products.

---

## For Technical Audience

### Project Structure
```
Weather_Forcast_Api/
├── app/
│	├── main.py                # FastAPI application with endpoints
│   └── src.py				# contains transformers used in main.py 
├── data/
│   ├── raw/               # Raw input data (e.g., weather_data.csv) used for testing purposes
│   └── processed/         # Processed data (if needed)
├── models/
│   ├── precipitation_fall/  # Regression models for precipitation
│   └── rain_or_not/         # Classification models for rain prediction
├── Api_testing.ipynb             # Example notebook for testing
├── Dockerhub.ipynb             # Example notebook for uploading project to docker
├── pyproject.toml             # Poetry project configuration
├── requirements.txt           # Python dependencies
├── poetry.lock                # Poetry lock file
└── readme.md                  # Project documentation
```


### How to Use the API
### Prerequisites
- Python ^3.11
- Poetry package manager


0. **Create Virtual Environment**
   ```bash
   # Create and activate a virtual Python environment with Python 3.11+
   python -m venv venv
   venv\Scripts\activate
   ```

1. **Install Poetry** (if not already installed):
   ```bash
   pip install poetry
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Run the API server:**
   ```bash
   poetry run fastapi dev app/main.py
   # Or alternatively:
   poetry run uvicorn app.main:app --reload
   ```

4. **Access the API:**
   - API Root: http://localhost:8000/
   - Health Check: http://localhost:8000/health/
   - Interactive Docs: http://localhost:8000/docs



#### Development Commands (Poetry)
```bash
# Enter Poetry shell
poetry shell

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Export requirements.txt (for Docker)
poetry export --without-hashes -f requirements.txt -o requirements.txt

# Run specific scripts
poetry run python script.py
```

3. **Send requests to the API endpoints** (see endpoint documentation below).
---


### API Endpoints


#### `/` (Root)
**Method:** GET

**Description:**
Displays a brief description of the project objectives, a list of available endpoints, all expected input parameter columns, output format of the model.

**Example:**
```
GET http://127.0.0.1:8000/
```
**Response Example:**
```json
{
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
        "will_rain": true
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
```


#### `/health/`
**Method:** GET

**Description:**
Returns status code 200 with a welcome message and model loading status.

**Example:**
```
GET http://127.0.0.1:8000/health/
```
**Response Example:**
```json
{
	"status": "healthy",
	"message": "Welcome to the Weather Prediction API! The service is running smoothly.",
	"models_loaded": {
		"rain_model": true,
		"precipitation_model": true
	}
}
```


#### `/predict/rain/`
**Method:** GET

**Description:**
Returns the prediction on if it will rain in exactly 7 days from the provided date. Accepts all weather parameters as optional query parameters (see expected_input_columns above).

**Input Parameters:**
- `date` (required): Date from which the model will predict rain or not in a week's time (format: YYYY-MM-DD)
- All other weather parameters are optional and will use defaults if not provided.

**Example:**
```
GET http://127.0.0.1:8000/predict/rain/?date=2023-01-01
```
**Response Example:**
```json
{
	"input_date": "2023-01-01",
	"prediction": {
		"date": "2023-01-08",
		"will_rain": true
	},
	"input_parameters": {
		"temperature_2m_mean": 17.6,
		"rain_sum": 0.1
		// ...other parameters if provided
	}
}
```

**Description:**
Returns the prediction on if it will rain in exactly 7 days from the provided date.

**Input Parameters:**
- `date`: Date from which the model will predict rain or not in a week's time (format: YYYY-MM-DD)

**Example:**
```
GET http://127.0.0.1:8000/predict/rain/?date=2023-01-01
```
**Response Example:**
```json
{
	"input_date": "2023-01-01",
	"prediction": {
		"date": "2023-01-08",
		"will_rain": true
	}
}
```


#### `/predict/precipitation/fall/`
**Method:** GET

**Description:**
Returns the predicted cumulated sum of precipitation (in mm) within the next 3 days from the provided date. Accepts all weather parameters as optional query parameters (see expected_input_columns above).

**Input Parameters:**
- `date` (required): Date from which the model will predict precipitation (format: YYYY-MM-DD)
- All other weather parameters are optional and will use defaults if not provided.

**Example:**
```
GET http://127.0.0.1:8000/predict/precipitation/fall/?date=2023-01-01
```
**Response Example:**
```json
{
	"input_date": "2023-01-01",
	"prediction": {
		"start_date": "2023-01-02",
		"end_date": "2023-01-04",
		"precipitation_fall": "28.2"
	},
	"input_parameters": {
		"temperature_2m_mean": 17.6,
		"rain_sum": 0.1
		// ...other parameters if provided
	}
}
```

#### `/docs`
**Method:** GET

**Description:**
Interactive API documentation (Swagger UI) auto-generated by FastAPI. Use this to explore and test all available endpoints directly from your browser.

**Example:**
```
Open http://127.0.0.1:8000/docs in your browser
```

#### 4. `/predict_rain`
**Method:** POST

**Description:** Predicts whether it will rain or not (classification) based on input weather features.

**Request Body Example:**
```json
{
	"Temperature": 25.0,
	"Humidity": 60.0,
	"WindDirection": "N",
	...
}
```

**Response Example:**
```json
{
	"rain": true
}
```

### API Endpoints

#### 1. `/predict_precipitation`
**Method:** POST

**Description:** Predicts the amount of precipitation (regression) based on input weather features.

**Request Body Example:**
```json
{
	"Temperature": 25.0,
	"Humidity": 60.0,
	"WindDirection": "N",
	...
}
```

**Response Example:**
```json
{
	"precipitation": 3.2
}
```

#### 2. `/predict_rain`
**Method:** POST

**Description:** Predicts whether it will rain or not (classification) based on input weather features.

**Request Body Example:**
```json
{
	"Temperature": 25.0,
	"Humidity": 60.0,
	"WindDirection": "N",
	...
}
```

**Response Example:**
```json
{
	"rain": true
}
```

---

### Model Details
- Uses advanced ML models (CatBoost, LightGBM, XGBoost, GradientBoosting, GaussianNB) for both regression and classification.
- Custom feature engineering and robust one-hot encoding for categorical variables (e.g., wind direction).
- Handles both batch and single-row predictions reliably.

---

## Docker Deployment

### Prerequisites
- Docker installed on your system
- Git (to clone the repository)

### Quick Start with Docker

1. **Export dependencies from Poetry:**
   ```bash
   poetry export --without-hashes -f requirements.txt -o requirements.txt
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t weather-prediction-api .
   ```

3. **Run the container:**
   ```bash
   docker run -d --name weather-api -p 8000:8000 weather-prediction-api
   ```

4. **Access the API:**
   - Test: Run Api_testing.ipynb
   - API: http://localhost:8000
   - Health Check: http://localhost:8000/health/
   - Documentation: http://localhost:8000/docs

### Docker Commands Reference

#### Build and Run
```bash
# Build the image
docker build --no-cache -t weather-prediction-api .

# Run container (detached mode)
docker run -d --name weather-api -p 8000:8000 weather-prediction-api

# Run container (interactive mode to see logs)
docker run --name weather-api -p 8000:8000 weather-prediction-api

# Check the startup logs
docker logs weather-api

```

#### Container Management
```bash
# Check running containers
docker ps

# Check container logs
docker logs weather-api

# Follow logs in real-time
docker logs -f weather-api

# Stop container
docker stop weather-api

# Remove container
docker rm weather-api

# Force remove running container
docker rm -f weather-api
```

#### Cleanup Commands
```bash
# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune

# Complete system cleanup (use with caution)
docker system prune
```

### Dockerfile Overview

The project includes a Dockerfile that:
- Uses Python 3.11-slim as base image
- Installs system dependencies (gcc for building packages)
- Installs Python dependencies from requirements.txt
- Copies the application code
- Exposes port 8000
- Runs the FastAPI application with uvicorn

### Testing the Dockerized API

Once the container is running, test the endpoints:

```bash
# Test health endpoint
curl http://localhost:8000/health/

# Test root endpoint
curl http://localhost:8000/

# Test rain prediction
curl "http://localhost:8000/predict/rain/?date=2023-01-01"

# Test precipitation prediction
curl "http://localhost:8000/predict/precipitation/fall/?date=2023-01-01"
```

### Troubleshooting

**Container won't start:**
- Check logs: `docker logs weather-api`
- Verify requirements.txt exists and contains uvicorn
- Ensure port 8000 is not in use: `netstat -an | findstr :8000`

**API not responding:**
- Verify container is running: `docker ps`
- Check if port mapping is correct: `-p 8000:8000`
- Test health endpoint first: `curl http://localhost:8000/health/`

**Rebuild after changes:**
```bash
docker rm -f weather-api
docker build -t weather-prediction-api .
docker run -d --name weather-api -p 8000:8000 weather-prediction-api
```

### Production Deployment

For production environments, consider:
- Using environment variables for configuration
- Setting up reverse proxy (nginx)
- Implementing logging and monitoring
- Using Docker Compose for multi-container setups
- Setting resource limits and health checks

---

## Docker Hub & Render Deployment

### Docker Hub
- The API Docker image is pushed to Docker Hub for easy sharing and deployment.
- To pull the image:
  ```bash
  docker pull doombuoyz/weather-prediction-api:latest
  ```

### Render Integration
- The Docker image is integrated and deployed on [Render](https://render.com).
- Live API URL: [https://weather-prediction-api-isp4.onrender.com](https://weather-prediction-api-isp4.onrender.com)
- You can access the API and docs directly at this link.

---

## Contribution & Support
- For technical questions, contact the development team.
- For business/HR inquiries, contact the project manager.

---

## License
This project is for internal use. Contact the project owner for licensing details.

