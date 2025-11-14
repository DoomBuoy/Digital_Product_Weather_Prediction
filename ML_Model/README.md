# ml_model

## Description
- What it is: A machine‑learning prototype that forecasts rainfall so teams can plan ahead.
- What it delivers:
  - rain_or_not notebook- Rain or Not (next week): Yes/No if rain is expected sometime next week.
  - precipitation_fall notebook- Rain Amount (short term): Estimated total rainfall (in millimeters) over the coming days.
- Who benefits: Operations, logistics, agriculture, events, and any team scheduling outdoor work.
- Why it matters: Reduces weather‑related delays, improves safety planning, and optimizes resource use.
- What it uses: Public weather readings (temperature, humidity, wind, cloud cover, sunshine, etc.).
- How reliable: Evaluated with standard metrics; geared to reduce missed rainy periods.
- Status: Working prototype with saved models and notebooks; ready to be wrapped into an API or app.

## Technical Overview

- Problem framing
  - rain_or_not (classification): Predicts if any rainfall occurs in the upcoming week for a given time/location window.
  - precipitation_fall (regression): Estimates short‑term cumulative precipitation in millimeters.

- Repository layout (code paths of interest)
  - ml_model/dataset.py: Downloads/prepares raw data into data/{raw,interim,processed}.
  - notebooks/: Exploratory, feature engineering, and end‑to‑end workflows.

- Data and features
  - Inputs: time‑stamped weather observations (temperature, humidity, wind speed/direction, cloud cover, sunshine/solar, etc.).
  - Preprocessing: type casting, handling missing/outlier values, time‑aware splits to avoid leakage.
  - Feature examples: wind direction categories, simple interactions (e.g., temp×humidity), and aggregated statistics; see notebooks for exact transformations.

- Modeling and evaluation
  - Tooling: scikit‑learn style estimators with reproducible seeds.
  - Validation: time‑based split/backtesting.
  - Metrics: classification — recall/precision/F1 (recall emphasized); regression — MAE/RMSE (and R²).
  - Artifacts: serialized models and preprocessing objects in models/; prepared datasets in data/processed.

## How to Run (Windows with Poetry)

### Prerequisites
- Python ^3.11
- Poetry package manager

### Setup Instructions

1. **Create Virtual Environment**
   ```bash
   # Create and activate a virtual Python environment with Python 3.11+
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   # Install project dependencies using Poetry
   poetry install
   ```

3. **Data Preparation**
   ```bash
   # Fetch and prepare the dataset and install poetry and other dependencies
   python ml_model/dataset.py
   ```

4. **Launch Jupyter Lab**
   ```bash
   # Start Jupyter Lab environment
   poetry run jupyter lab
   ```

5. **Run Experiments**
   - Open and run `notebooks/rain_or_not/` for binary rain prediction
   - Open and run `notebooks/precipitation_fall/` for rainfall amount prediction
   - Trained models and processed data will be saved to `models/` and `data/processed/`

### Quick Start
```bash
# One-line setup (after creating virtual environment)
poetry install && python ml_model/dataset.py && poetry run jupyter lab
```

### Output Locations
- **Trained Models**: `models/`
- **Processed Data**: `data/processed/`
- **Results**: Available in notebook outputs and saved artifacts
- API readiness
  - ml_model/modeling/predict.py can be wrapped in a FastAPI/Flask endpoint to serve forecasts from saved models.

- Notes
  - Exact model choices, hyperparameters, and feature sets are documented in the notebooks; refer there for experiment details and results.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│   └── rain_or_not     <- notebook for rain or not prediction model
│   └── precipitation_fall   <- notebook for precipitation fall prediction                 
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ml_model and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ml_model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ml_model a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

