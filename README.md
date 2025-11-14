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

## 📖 Project Descriptions

### Non-Technical Description (For HR)
View this product at - https://digitalappuctweatherprediction-tfdvs8zextpi3xvktbpogy.streamlit.app/
This weather prediction digital product is an innovative tool that helps businesses and individuals make informed decisions based on accurate weather forecasts. It predicts whether it will rain in exactly 7 days and estimates cumulative precipitation over the next 3 days, using advanced machine learning algorithms. The system includes a user-friendly web application for easy access, a reliable API for seamless integration into other systems, and is deployed in the cloud for 24/7 availability. It's particularly valuable for industries such as agriculture (planning irrigation and harvests), logistics (optimizing routes and schedules), event management (weather contingency planning), and operations management (reducing weather-related risks and delays). By providing reliable weather insights, this product helps organizations minimize uncertainties, improve planning, and enhance operational efficiency.

### Technical Description (For Interviewer)
This project implements a comprehensive weather prediction system using ensemble machine learning techniques. The architecture follows a modular design with three main components: an ML pipeline for model development and evaluation, a FastAPI-based REST API for serving predictions, and a Streamlit web application for user interaction. The ML models employ ensemble methods combining CatBoost, XGBoost, LightGBM, GradientBoosting, and GaussianNB classifiers, trained on historical weather data with over 50 features including temperature, humidity, wind patterns, cloud cover, and soil moisture. The system integrates real-time weather data from the Open-Meteo API and uses time-based cross-validation to prevent data leakage. Key technical challenges addressed include handling time-series data, optimizing for high recall in rain prediction to minimize false negatives, implementing robust error handling for missing values, and ensuring scalable deployment through Docker containerization. The API is hosted on Render with automatic health checks and Swagger documentation, while the web app runs on Streamlit Cloud, demonstrating full-stack development from data science to production deployment.

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
├── ML_Model/              # Machine Learning Pipeline for the product
│   
├── Weather_Forcast_Api/   # REST API Service creatred for ML model deployment
│   
└── App/                   # Web Application with user frindly usage.
    └── App_main.py        # Streamlit interface
```

## 📊 Model Details

### Rain Prediction Model
- **Type**: Binary Classification
- **Target**: Will it rain in exactly next 7 days?
- **Algorithm**: Ensemble of CatBoost, XGBoost, LightGBM, GradientBoosting, GaussianNB


### Precipitation Prediction Model
- **Type**: Regression
- **Target**: Cumulative precipitation (mm) over next 3 days
- **Algorithm**: Ensemble of advanced ML models


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
Visit the deployed Streamlit app: [Weather Prediction App](https://digitalappuctweatherprediction-tfdvs8zextpi3xvktbpogy.streamlit.app/)

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
- **Live App**: https://digitalappuctweatherprediction-tfdvs8zextpi3xvktbpogy.streamlit.app/
- **API Docs**: https://weather-prediction-api-isp4.onrender.com/docs
- **Docker Hub**: https://hub.docker.com/r/doombuoyz/weather-prediction-api

---

*Built with ❤️ using Python, FastAPI, Streamlit, and advanced machine learning techniques.*