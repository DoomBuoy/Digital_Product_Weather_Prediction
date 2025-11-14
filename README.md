# Weather Prediction Digital Product

A comprehensive weather prediction system that provides accurate forecasts for rain occurrence and precipitation amounts using advanced machine learning models. The system includes machine learning model training, a production-ready API, and an interactive web application.

## 🌟 Overview

This digital product delivers two key weather prediction capabilities:
- **Rain Prediction**: Binary classification to predict if it will rain exactly 7 days from a given date
- **Precipitation Prediction**: Regression model to forecast cumulative precipitation (in mm) over the next 3 days

The system is built with a modular architecture consisting of three main components:
- **ML_Model**: Machine learning pipeline for model training and evaluation
- **Weather_Forcast_Api**: FastAPI-based REST API for serving predictions
- **App**: Streamlit web application for user interaction

## 🚀 Key Features

- **Accurate Predictions**: Uses ensemble ML models (CatBoost, XGBoost, LightGBM) trained on comprehensive weather data
- **Real-time Data Integration**: Fetches live weather data from Open-Meteo API
- **Production-Ready API**: FastAPI backend with automatic documentation and health checks
- **User-Friendly Interface**: Streamlit app with real-time and custom input options
- **Docker Support**: Containerized deployment for easy scaling
- **Cloud Deployment**: Live API hosted on Render with 24/7 availability

## 🏗️ Architecture

```
Weather_Prediction_Digital_Product/
├── ML_Model/              # Machine Learning Pipeline
│   ├── notebooks/         # Jupyter notebooks for experiments
│   ├── ml_model/          # Source code for data processing and modeling
│   ├── models/            # Trained model artifacts
│   └── data/              # Raw and processed datasets
├── Weather_Forcast_Api/   # REST API Service
│   ├── app/               # FastAPI application
│   ├── models/            # Serialized models for inference
│   └── Dockerfile         # Container configuration
└── App/                   # Web Application
    └── App_main.py        # Streamlit interface
```

## 📊 Model Details

### Rain Prediction Model
- **Type**: Binary Classification
- **Target**: Will it rain in exactly 7 days?
- **Algorithm**: Ensemble of CatBoost, XGBoost, LightGBM, GradientBoosting, GaussianNB
- **Evaluation**: Focus on recall to minimize missed rainy periods

### Precipitation Prediction Model
- **Type**: Regression
- **Target**: Cumulative precipitation (mm) over next 3 days
- **Algorithm**: Ensemble of advanced ML models
- **Evaluation**: MAE/RMSE metrics for accuracy assessment

## 🔧 Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- Docker (optional, for containerized deployment)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Weather_Prediction_Digital_Product
```

### 2. Try the Live Application
Visit the deployed Streamlit app: [Weather Prediction App](https://weather-prediction-app.streamlit.app/)

Or access the live API directly: [Weather Prediction API](https://weather-prediction-api-isp4.onrender.com)

### 3. Local Development

#### API Setup
```bash
cd Weather_Forcast_Api
poetry install
poetry run fastapi dev app/main.py
```

#### Web App Setup
```bash
cd App
poetry install
poetry run streamlit run App_main.py
```

#### ML Pipeline Setup
```bash
cd ML_Model
poetry install
poetry run jupyter lab
```

## 🐳 Docker Deployment

### API Container
```bash
cd Weather_Forcast_Api
docker build -t weather-prediction-api .
docker run -p 8000:8000 weather-prediction-api
```

### Pull from Docker Hub
```bash
docker pull doombuoyz/weather-prediction-api:latest
docker run -p 8000:8000 doombuoyz/weather-prediction-api:latest
```

## 📚 API Documentation

The API provides the following endpoints:

- `GET /` - Project overview and endpoint documentation
- `GET /health/` - Health check and model loading status
- `GET /docs` - Interactive Swagger UI documentation
- `GET /predict/rain/` - Rain prediction (7 days ahead)
- `GET /predict/precipitation/fall/` - Precipitation prediction (next 3 days)

### Example API Usage
```bash
# Health check
curl https://weather-prediction-api-isp4.onrender.com/health/

# Rain prediction
curl "https://weather-prediction-api-isp4.onrender.com/predict/rain/?date=2023-01-01"

# Precipitation prediction
curl "https://weather-prediction-api-isp4.onrender.com/predict/precipitation/fall/?date=2023-01-01"
```

## 🎯 Use Cases

- **Operations Planning**: Help teams schedule outdoor activities and prepare for weather-dependent operations
- **Agriculture**: Assist farmers in planning irrigation and harvest activities
- **Event Management**: Support event planners in making weather contingency decisions
- **Logistics**: Enable better route planning and resource allocation
- **Risk Management**: Reduce weather-related delays and safety concerns

## 📈 Data Sources

- **Training Data**: Historical weather observations from public sources
- **Real-time Data**: Open-Meteo API for current weather conditions
- **Features**: 50+ weather parameters including temperature, humidity, wind, cloud cover, soil moisture, etc.

## 🔍 Model Performance

- **Rain Prediction**: Optimized for high recall to minimize false negatives
- **Precipitation Prediction**: Strong correlation with actual measurements
- **Validation**: Time-based cross-validation to prevent data leakage
- **Robustness**: Handles missing values and outliers gracefully

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for internal use. Contact the project owner for licensing details.

## 📞 Support

- **Technical Issues**: Check the component-specific READMEs in each folder
- **API Documentation**: Visit `/docs` endpoint when API is running
- **Model Details**: Refer to notebooks in `ML_Model/notebooks/`

## 🔗 Links

- **Live API**: https://weather-prediction-api-isp4.onrender.com
- **Live App**: https://weather-prediction-app.streamlit.app/
- **API Docs**: https://weather-prediction-api-isp4.onrender.com/docs
- **Docker Hub**: https://hub.docker.com/r/doombuoyz/weather-prediction-api

---

*Built with ❤️ using Python, FastAPI, Streamlit, and advanced machine learning techniques.*