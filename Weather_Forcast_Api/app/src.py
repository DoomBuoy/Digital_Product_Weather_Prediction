# <Student to fill this section>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from scipy import stats as st
import os
import joblib
import math
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.metrics import f1_score
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import os
# # Read all three saved models
import joblib
import os
import traceback
from sklearn.ensemble import VotingClassifier
# Do not modify this code
import warnings
warnings.simplefilter(action='ignore')

# 1. Rain Binary Transformer
class RainBinaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, shift_days=-7):
        self.shift_days = shift_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'rain_sum' not in X.columns:
            raise KeyError("Column 'rain_sum' not found in input data.")

        # Create the rain_binary column
        X['rain_binary'] = (X['rain_sum'] > 0).astype(int)

        # Shift the rain_binary column
        X['rain_binary'] = X['rain_binary'].shift(self.shift_days)

        # Drop rows with NaN values in the rain_binary column
        # X.dropna(subset=['rain_binary'], inplace=True)    #not for prediction
        X.fillna(0, inplace=True)  # For prediction, fill NaNs with 0

        # Ensure the rain_binary column is integer type
        X['rain_binary'] = X['rain_binary'].astype(int)

        return X

# 2. Wind Direction Compass Transformer
class WindDirectionCompassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_col='wind_direction_10m_dominant', 
                 numeric_col='wind_direction_10m_dominant_deg',
                 compass_col='wind_dir_compass'):
        self.input_col = input_col
        self.numeric_col = numeric_col
        self.compass_col = compass_col
        self.dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                    'S','SSW','SW','WSW','W','WNW','NW','NNW']

    def deg_to_compass(self, deg):
        """Convert degrees to 16-point compass direction"""
        try:
            deg = float(deg)
            if np.isnan(deg):
                return np.nan
        except (ValueError, TypeError):
            return np.nan
        
        idx = int(((deg + 11.25) % 360) / 22.5)
        return self.dirs[idx]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        if self.input_col not in X.columns:
            raise KeyError(f"Column '{self.input_col}' not found in input data.")
        
        # Convert to numeric (handles strings/missing values)
        X[self.numeric_col] = pd.to_numeric(X[self.input_col], errors='coerce')
        
        # Apply compass conversion
        X[self.compass_col] = X[self.numeric_col].apply(
            lambda x: self.deg_to_compass(x) if pd.notna(x) else np.nan
        )
        
        return X

# 3. Drop Columns Transformer
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns_to_drop, errors='ignore')

# 4. Cap Outliers Using Bounds Transformer
class CapOutliersUsingBounds(BaseEstimator, TransformerMixin):
    def __init__(self, cap_stats):
        self.cap_stats = cap_stats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, stats in self.cap_stats.items():
            if col in X.columns:
                lower_fill = stats['lower_fill']
                upper_fill = stats['upper_fill']

                X[col] = pd.to_numeric(X[col], errors='coerce')
                X.loc[X[col] < lower_fill, col] = lower_fill
                X.loc[X[col] > upper_fill, col] = upper_fill
        return X

# 5. Temperature Humidity Interaction Transformer
class TempHumidityInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, temp_col='temperature_2m_min', humidity_col='temperature_2m_mean', 
                 interaction_col='temp_humidity_interaction'):
        self.temp_col = temp_col
        self.humidity_col = humidity_col
        self.interaction_col = interaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.temp_col not in X.columns or self.humidity_col not in X.columns:
            raise KeyError(f"Columns '{self.temp_col}' and/or '{self.humidity_col}' not found in input data.")

        # Create the interaction feature
        X[self.interaction_col] = X[self.temp_col] * X[self.humidity_col]
        return X

# 6. Cloud Sun Interaction Transformer
class CloudSunInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cloud_col='shortwave_radiation_sum', sun_col='daylight_duration', 
                 interaction_col='solar_radiation_interaction'):
        self.cloud_col = cloud_col
        self.sun_col = sun_col
        self.interaction_col = interaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.cloud_col not in X.columns or self.sun_col not in X.columns:
            raise KeyError(f"Columns '{self.cloud_col}' and/or '{self.sun_col}' not found in input data.")

        # Create the interaction feature
        X[self.interaction_col] = X[self.cloud_col] * X[self.sun_col]
        return X

# 7. Radiation Per Sunhour Transformer (Dew Bulb Interaction)
class RadiationPerSunhourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, radiation_col='dew_point_2m_mean', sun_col='wet_bulb_temperature_2m_mean', 
                 interaction_col='dew_bulb_interation'):
        self.radiation_col = radiation_col
        self.sun_col = sun_col
        self.interaction_col = interaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.radiation_col not in X.columns or self.sun_col not in X.columns:
            raise KeyError(f"Columns '{self.radiation_col}' and/or '{self.sun_col}' not found in input data.")

        # Avoid division by zero
        X[self.interaction_col] = X[self.radiation_col] / (X[self.sun_col].replace(0, 1e-6))
        return X

# 8. Log1p Transformer
class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_transform):
        self.columns_to_transform = columns_to_transform
        self.transform_info = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        self.transform_info = {}
        
        for col in self.columns_to_transform:
            if col not in X.columns:
                continue
                
            # Coerce to numeric if needed
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = pd.to_numeric(X[col], errors='coerce')

            col_min = X[col].min(skipna=True)
            if pd.isna(col_min):
                print(f"Skipping {col}: all values NaN")
                continue

            shift = 0.0
            if col_min <= -1.0:
                shift = float(abs(col_min) + 1e-6)
                X[col] = X[col] + shift  # Make values > -1

            # Apply log1p (safe for zeros and values > -1)
            X[col] = np.log1p(X[col])
            self.transform_info[col] = {'shift_added': shift}

        return X

# 9. One Hot Encode Transformer
class OneHotEncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_encode, prefix=None, drop_first=False, all_possibilities=None):
        self.column_to_encode = column_to_encode
        self.prefix = prefix
        self.drop_first = drop_first
        self.columns_ = None
        self.all_possibilities = all_possibilities  # List of all possible categories

    def fit(self, X, y=None):
        if self.column_to_encode not in X.columns:
            raise KeyError(f"Column '{self.column_to_encode}' not found in input data.")
        # Use all_possibilities if provided, else infer from data
        if self.all_possibilities is not None:
            categories = self.all_possibilities
        else:
            categories = X[self.column_to_encode].unique()
        # Create a dummy DataFrame to get all columns
        dummy_df = pd.DataFrame({self.column_to_encode: categories})
        encoded = pd.get_dummies(dummy_df, prefix=self.prefix, drop_first=self.drop_first)
        self.columns_ = encoded.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        if self.column_to_encode not in X.columns:
            raise KeyError(f"Column '{self.column_to_encode}' not found in input data.")
        # Perform one-hot encoding
        encoded = pd.get_dummies(X[self.column_to_encode], prefix=self.prefix, drop_first=self.drop_first)
        # Ensure all expected columns are present (add missing ones with 0s)
        for col in self.columns_:
            if col not in encoded.columns:
                encoded[col] = 0
        # Reorder columns to match training order
        encoded = encoded[self.columns_]
        X = pd.concat([X, encoded], axis=1)
        X.drop(columns=[self.column_to_encode], inplace=True)
        return X

# 10. MinMax Scaler Transformer
class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='rain_binary'):
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.numeric_features = None

    def fit(self, X, y=None):
        # Select numeric columns excluding the target column
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.drop(self.target_column, errors='ignore')
        self.scaler.fit(X[self.numeric_features])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        return X

# 11. Dataset Splitter Transformer
class DatasetSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, train_size=0.7, val_size=0.15):
        self.target_column = target_column
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = 1 - train_size - val_size
        self.splits = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure data is sorted by time if 'date' column exists
        if 'date' in X.columns:
            X = X.sort_values(by='date', ascending=True)
            X.drop(columns=['date'], inplace=True, errors='ignore')

        # Define split indices
        train_size = int(len(X) * self.train_size)
        val_size = int(len(X) * self.val_size)

        # Split data
        train = X.iloc[:train_size]
        val = X.iloc[train_size:train_size + val_size]
        test = X.iloc[train_size + val_size:]

        # Separate features and target
        self.splits = {
            'X_train': train.drop(columns=[self.target_column]),
            'y_train': train[self.target_column],
            'X_val': val.drop(columns=[self.target_column]),
            'y_val': val[self.target_column],
            'X_test': test.drop(columns=[self.target_column]),
            'y_test': test[self.target_column]
        }
        return self.splits

# 12. Preprocess and Extract Features Transformer
class PreprocessAndExtractFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessing_pipeline, target_column='rain_binary'):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.target_column = target_column
    
    def fit(self, X, y=None):
        self.preprocessing_pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        processed_data = self.preprocessing_pipeline.transform(X)
        return processed_data

# 13. Loaded Model Transformer
class LoadedModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # For consistency with pipeline, return predictions as DataFrame
        predictions = self.model.predict(X)
        return pd.DataFrame({'predictions': predictions}, index=X.index)
    
    def predict(self, X):
        return self.model.predict(X)

print("All transformers defined successfully!")


# Load all three models
lgbm_model_path = r"models/rain_or_not/lgbm_best_model.pkl"
xgb_model_path = r"models/rain_or_not/xgb_best_model.pkl"
gnb_model_path = r"models/rain_or_not/gnb_best_model.pkl"

lgbm_loaded_model = joblib.load(lgbm_model_path)
xgb_loaded_model = joblib.load(xgb_model_path)
gnb_loaded_model = joblib.load(gnb_model_path)

print("All three models loaded successfully!")

# Define the missing transformer classes
class PreprocessAndExtractFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessing_pipeline, target_column='rain_binary'):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.target_column = target_column
    
    def fit(self, X, y=None):
        self.preprocessing_pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        processed_data = self.preprocessing_pipeline.transform(X)
        return processed_data

class LoadedModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # For consistency with pipeline, return predictions as DataFrame
        predictions = self.model.predict(X)
        return pd.DataFrame({'predictions': predictions}, index=X.index)
    
    def predict(self, X):
        return self.model.predict(X)

# Create the comprehensive preprocessing pipeline from your notebook
preprocessing_pipeline = Pipeline(steps=[
    ('rain_binary_transformer', RainBinaryTransformer(shift_days=-7)),
    ('wind_direction_compass', WindDirectionCompassTransformer(
        input_col='wind_direction_10m_dominant',
        numeric_col='wind_direction_10m_dominant_deg',
        compass_col='wind_dir_compass'
    )),
    ('drop_leakage_columns', DropColumnsTransformer(columns_to_drop=[
        'weather_code', 'snowfall_sum', 'snowfall_water_equivalent_sum',
        'precipitation_sum', 'precipitation_hours', 'rain_sum'
    ])),
    ('drop_low_corr', DropColumnsTransformer(columns_to_drop=[
        'wind_speed_10m_max', 'wind_gusts_10m_max', 'sunshine_duration', 
        'cloud_cover_mean', 'cloud_cover_max', 'cloud_cover_min', 
        'relative_humidity_2m_mean', 'relative_humidity_2m_max', 
        'relative_humidity_2m_min', 'pressure_msl_mean', 'pressure_msl_max', 
        'pressure_msl_min', 'surface_pressure_mean', 'surface_pressure_max', 
        'surface_pressure_min', 'wind_gusts_10m_mean', 'wind_speed_10m_mean', 
        'wind_gusts_10m_min', 'wind_speed_10m_min', 'vapour_pressure_deficit_max', 
        'soil_moisture_0_to_100cm_mean', 'soil_moisture_0_to_7cm_mean', 
        'soil_moisture_28_to_100cm_mean', 'soil_moisture_7_to_28cm_mean'
    ])),
    ('drop_time_columns', DropColumnsTransformer(columns_to_drop=['month', 'hour'])),
    ('cap_outliers', CapOutliersUsingBounds(cap_stats={
        'temperature_2m_mean': {'lower_fill': 3.4995476874999962, 'upper_fill': 31.404034187500006},
        'temperature_2m_max': {'lower_fill': 7.9326224999999955, 'upper_fill': 34.4336265},
        'temperature_2m_min': {'lower_fill': 0.0, 'upper_fill': 29.1455},
        'apparent_temperature_mean': {'lower_fill': 0.0, 'upper_fill': 35.843287375},
        'apparent_temperature_max': {'lower_fill': 1.541537749999998, 'upper_fill': 40.68207775},
        'apparent_temperature_min': {'lower_fill': 0.0, 'upper_fill': 33.3258332125},
        'wind_direction_10m_dominant': {'lower_fill': 0.0, 'upper_fill': 555.4815375000001},
        'shortwave_radiation_sum': {'lower_fill': 0.0, 'upper_fill': 38.78},
        'et0_fao_evapotranspiration': {'lower_fill': 0.0, 'upper_fill': 7.5502130375},
        'sunrise': {'lower_fill': 1025938170.375, 'upper_fill': 1971334109.375},
        'sunset': {'lower_fill': 1025985350.125, 'upper_fill': 1971374281.125},
        'daylight_duration': {'lower_fill': 21837.203, 'upper_fill': 65289.165},
        'dew_point_2m_mean': {'lower_fill': 0.0, 'upper_fill': 28.124244},
        'dew_point_2m_max': {'lower_fill': 0.045000000000001705, 'upper_fill': 28.912999999999997},
        'dew_point_2m_min': {'lower_fill': 0.0, 'upper_fill': 27.4879975},
        'et0_fao_evapotranspiration_sum': {'lower_fill': 0.0, 'upper_fill': 7.5502130375},
        'winddirection_10m_dominant': {'lower_fill': 0.0, 'upper_fill': 555.4815375000001},
        'wet_bulb_temperature_2m_mean': {'lower_fill': 0.14711612499999838, 'upper_fill': 28.310293125},
        'wet_bulb_temperature_2m_max': {'lower_fill': 3.0461678750000036, 'upper_fill': 29.143022874999993},
        'wet_bulb_temperature_2m_min': {'lower_fill': 0.0, 'upper_fill': 27.61814225},
        'soil_temperature_0_to_100cm_mean': {'lower_fill': 4.280321750000001, 'upper_fill': 32.05357575},
        'soil_temperature_0_to_7cm_mean': {'lower_fill': 2.105079874999996, 'upper_fill': 34.300586875},
        'soil_temperature_7_to_28cm_mean': {'lower_fill': 2.8547588750000017, 'upper_fill': 33.474513875},
        'soil_temperature_28_to_100cm_mean': {'lower_fill': 4.921099875000001, 'upper_fill': 31.310672875},
        'wind_direction_10m_dominant_deg': {'lower_fill': 0.0, 'upper_fill': 555.4815375000001}
    })),
    ('temp_humidity_interaction', TempHumidityInteractionTransformer(
        temp_col='temperature_2m_min',
        humidity_col='temperature_2m_mean',
        interaction_col='temp_humidity_interaction'
    )),
    ('solar_radiation_interaction', CloudSunInteractionTransformer(
        cloud_col='shortwave_radiation_sum',
        sun_col='daylight_duration',
        interaction_col='solar_radiation_interaction'
    )),
    ('dew_bulb_interaction', RadiationPerSunhourTransformer(
        radiation_col='dew_point_2m_mean',
        sun_col='wet_bulb_temperature_2m_mean',
        interaction_col='dew_bulb_interation'
    )),
    ('log1p_transform', Log1pTransformer(columns_to_transform=[
        'temperature_2m_max', 'et0_fao_evapotranspiration', 
        'et0_fao_evapotranspiration_sum', 'solar_radiation_interaction'
    ])),
    ('drop_wind_speed_bin', DropColumnsTransformer(columns_to_drop=['wind_speed_bin'])),

    ('one_hot_encode', OneHotEncodeTransformer(
                column_to_encode='wind_dir_compass',
                prefix='wind_dir',
                drop_first=False,
                all_possibilities=[
            'N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW'])),
            # ...other steps...
    
    ('minmax_scaler', MinMaxScalerTransformer(target_column='rain_binary')),
    ('dataset_splitter', DatasetSplitter(
        target_column='rain_binary', 
        train_size=0.7, 
        val_size=0.15
    ))
])

# Updated models dictionary with all three models
models_dict = {
    'lgbm': lgbm_loaded_model,
    'xgb': xgb_loaded_model,
    'gnb': gnb_loaded_model
}

# Enhanced Multi-Model Pipeline with all three models
class MultiModelPredictionPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessing_pipeline, models_dict, target_column='rain_binary'):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.models_dict = models_dict
        self.target_column = target_column
    
    def fit(self, X, y=None):
        # Fit preprocessing pipeline
        processed_data = self.preprocessing_pipeline.fit_transform(X)
        return self
    
    def predict(self, X, model_name=None, voting_method='majority'):
        """
        Make predictions using specified model or ensemble
        model_name: 'lgbm', 'xgb', 'gnb', or None (defaults to ensemble)
        voting_method: 'majority', 'weighted', or 'soft' for ensemble predictions
        """
        # Apply preprocessing
        processed_splits = self.preprocessing_pipeline.transform(X)
        
        # Extract features from the splits (use X_train as features)
        if isinstance(processed_splits, dict) and 'X_train' in processed_splits:
            # Combine all splits for prediction
            features = pd.concat([
                processed_splits['X_train'], 
                processed_splits['X_val'], 
                processed_splits['X_test']
            ], ignore_index=True)
        else:
            # Extract features by dropping target column
            if self.target_column in processed_splits.columns:
                features = processed_splits.drop(columns=[self.target_column])
            else:
                features = processed_splits
        
        if model_name and model_name in self.models_dict:
            # Use specific model
            predictions = self.models_dict[model_name].predict(features)
        else:
            # Use ensemble prediction
            individual_preds = []
            for name, model in self.models_dict.items():
                pred = model.predict(features)
                individual_preds.append(pred)
            
            individual_preds = np.array(individual_preds)
            
            if voting_method == 'majority':
                # Simple majority voting
                predictions = np.round(np.mean(individual_preds, axis=0)).astype(int)
            elif voting_method == 'weighted':
                # Weighted voting based on model performance
                weights = np.array([0.4, 0.25, 0.35])  # lgbm, xgb, gnb
                weighted_preds = np.average(individual_preds, weights=weights, axis=0)
                predictions = np.round(weighted_preds).astype(int)
            elif voting_method == 'soft':
                # Soft voting using probabilities
                individual_probas = []
                for model in self.models_dict.values():
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)[:, 1]  # Get probability of class 1
                        individual_probas.append(proba)
                
                if individual_probas:
                    avg_probas = np.mean(individual_probas, axis=0)
                    predictions = (avg_probas > 0.5).astype(int)
                else:
                    # Fallback to majority voting
                    predictions = np.round(np.mean(individual_preds, axis=0)).astype(int)
        
        return predictions
    
    def predict_proba(self, X, model_name=None):
        """
        Get prediction probabilities
        """
        # Apply preprocessing
        processed_splits = self.preprocessing_pipeline.transform(X)
        
        # Extract features
        if isinstance(processed_splits, dict) and 'X_train' in processed_splits:
            features = pd.concat([
                processed_splits['X_train'], 
                processed_splits['X_val'], 
                processed_splits['X_test']
            ], ignore_index=True)
        else:
            if self.target_column in processed_splits.columns:
                features = processed_splits.drop(columns=[self.target_column])
            else:
                features = processed_splits
        
        if model_name and model_name in self.models_dict:
            # Use specific model
            if hasattr(self.models_dict[model_name], 'predict_proba'):
                probabilities = self.models_dict[model_name].predict_proba(features)
            else:
                return None
        else:
            # Use ensemble (average probabilities)
            individual_probas = []
            for model in self.models_dict.values():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)
                    individual_probas.append(proba)
            
            if individual_probas:
                probabilities = np.mean(individual_probas, axis=0)
            else:
                return None
        
        return probabilities
    
    def get_individual_predictions(self, X):
        """
        Get predictions from all individual models
        """
        # Apply preprocessing
        processed_splits = self.preprocessing_pipeline.transform(X)
        
        # Extract features
        if isinstance(processed_splits, dict) and 'X_train' in processed_splits:
            features = pd.concat([
                processed_splits['X_train'], 
                processed_splits['X_val'], 
                processed_splits['X_test']
            ], ignore_index=True)
        else:
            if self.target_column in processed_splits.columns:
                features = processed_splits.drop(columns=[self.target_column])
            else:
                features = processed_splits
        
        results = {}
        for name, model in self.models_dict.items():
            results[name] = model.predict(features)
        
        return results
    
    def get_model_performance_summary(self, X, y_true):
        """
        Get performance summary for all models
        """
        individual_preds = self.get_individual_predictions(X)
        
        performance = {}
        for name, predictions in individual_preds.items():
            recall = recall_score(y_true, predictions)
            f1 = f1_score(y_true, predictions)
            performance[name] = {
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy_score(y_true, predictions)
            }
        
        # Add ensemble performance
        ensemble_pred = self.predict(X, voting_method='majority')
        performance['ensemble_majority'] = {
            'recall': recall_score(y_true, ensemble_pred),
            'f1_score': f1_score(y_true, ensemble_pred),
            'accuracy': accuracy_score(y_true, ensemble_pred)
        }
        
        return performance

# Create the enhanced multi-model prediction pipeline
multi_model_pipeline = MultiModelPredictionPipeline(
    preprocessing_pipeline=preprocessing_pipeline,
    models_dict=models_dict,
    target_column='rain_binary'
)

# Test the pipeline
try:
    # Load fresh data for testing
    test_data = pd.read_csv(r'data/raw/weather_data.csv')
    print("\n")
    print("=== Testing Rain_or_not Pipeline with All Three Models ===")
    print("\n")
    # Test multi-model pipeline
    multi_model_pipeline.fit(test_data)
    
    # Get ensemble predictions with different voting methods
    majority_pred = multi_model_pipeline.predict(test_data, voting_method='majority')
    weighted_pred = multi_model_pipeline.predict(test_data, voting_method='weighted')
    soft_pred = multi_model_pipeline.predict(test_data, voting_method='soft')
    
    print(f"Majority voting predictions: {majority_pred[:10]}")
    print(f"Weighted voting predictions: {weighted_pred[:10]}")
    print(f"Soft voting predictions: {soft_pred[:10]}")
    
    # Get individual model predictions
    individual_preds = multi_model_pipeline.get_individual_predictions(test_data)
    print(f"\nIndividual predictions:")
    for name, preds in individual_preds.items():
        print(f"  {name}: {preds[:10]}")
    
    # Get specific model predictions
    lgbm_pred = multi_model_pipeline.predict(test_data, model_name='lgbm')
    xgb_pred = multi_model_pipeline.predict(test_data, model_name='xgb')
    gnb_pred = multi_model_pipeline.predict(test_data, model_name='gnb')
    
    print(f"\nSpecific model predictions:")
    print(f"LightGBM: {lgbm_pred[:10]}")
    print(f"XGBoost: {xgb_pred[:10]}")
    print(f"GaussianNB: {gnb_pred[:10]}")
    
    # Get ensemble probabilities
    ensemble_proba = multi_model_pipeline.predict_proba(test_data)
    if ensemble_proba is not None:
        print(f"\nEnsemble probabilities shape: {ensemble_proba.shape}")
        print(f"Sample probabilities: {ensemble_proba[:5]}")
    
except Exception as e:
    print(f"Error during testing: {e}")
    
    traceback.print_exc()

print("\nComplete pipeline with all three models created successfully!")

# Create a voting classifier for comparison
voting_classifier = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_loaded_model),
        ('xgb', xgb_loaded_model),
        ('gnb', gnb_loaded_model)
    ],
    voting='hard'  # Change to 'soft' for probability-based voting
)

# print("\nAvailable pipeline options:")
# print("1. multi_model_pipeline - Flexible pipeline with all three models and different voting methods")
# print("2. voting_classifier - Uses sklearn VotingClassifier")

print("\n")
print("===Test Successfull Rain_or_not Pipeline with All Three Models ===")
print("\n")
print("\n")
print("\n")



###############################################################################################################################################################

# Precipitation fall starts here

####################################################################################################################




# 1. Rolling Cumulative Sum Transformer
class RollingCumulativeSumTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name='precipitation_sum', new_column_name='precipitation_sum_next_4_days', window=4):
        self.column_name = column_name
        self.new_column_name = new_column_name
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.sort_values(by='date')  # Ensure DataFrame is sorted by date
        # Calculate rolling sum for the next 4 days (including today)
        X[self.new_column_name] = (
            X[self.column_name]
            .rolling(window=self.window, min_periods=1)  # Rolling sum over the window
            .sum()
            .shift(-(self.window - 1))  # Shift to align with the next 4 days
        ).round(1)  # Round to 1 decimal place
        return X

# 2. Remove Last Rows Transformer
class RemoveLastRowsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rows_to_remove=3):
        self.rows_to_remove = rows_to_remove

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Remove the last `rows_to_remove` rows
        X = X.iloc[:-self.rows_to_remove]
        return X

# 3. Wind Direction Compass Transformer
class WindDirectionCompassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_column='wind_direction_10m_dominant', 
                 degree_column='wind_direction_10m_dominant_deg', 
                 compass_column='wind_dir_compass'):
        self.input_column = input_column
        self.degree_column = degree_column
        self.compass_column = compass_column
        self.dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                     'S','SSW','SW','WSW','W','WNW','NW','NNW']

    def deg_to_compass(self, deg):
        """Convert degrees to 16-point compass direction"""
        try:
            deg = float(deg)
            if np.isnan(deg):
                return np.nan
        except (ValueError, TypeError):
            return np.nan
        
        idx = int(((deg + 11.25) % 360) / 22.5)
        return self.dirs[idx]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        if self.input_column not in X.columns:
            raise KeyError(f"Column '{self.input_column}' not found in input data.")
        
        # Convert to numeric (handles strings/missing values)
        X[self.degree_column] = pd.to_numeric(X[self.input_column], errors='coerce')
        
        # Apply compass conversion
        X[self.compass_column] = X[self.degree_column].apply(
            lambda x: self.deg_to_compass(x) if pd.notna(x) else np.nan
        )
        
        return X

# 4. Drop Columns Transformer
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns_to_drop, errors='ignore')

# 5. Cap Outliers From Dict Transformer
class CapOutliersFromDict(BaseEstimator, TransformerMixin):
    def __init__(self, bounds_dict):
        self.bounds_dict = bounds_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, bounds in self.bounds_dict.items():
            if col not in X.columns:
                continue
                
            lower_fill = bounds['lower_fill']
            upper_fill = bounds['upper_fill']

            X[col] = pd.to_numeric(X[col], errors='coerce')
            X.loc[X[col] < lower_fill, col] = lower_fill
            X.loc[X[col] > upper_fill, col] = upper_fill
        return X

# 6. Humidity Dew Point Interaction Transformer
class HumidityDewPointInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, humidity_col='relative_humidity_2m_mean', 
                 dewpoint_col='dew_point_2m_mean', 
                 interaction_col='humidity_dewpoint_interaction'):
        self.humidity_col = humidity_col
        self.dewpoint_col = dewpoint_col
        self.interaction_col = interaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.humidity_col not in X.columns or self.dewpoint_col not in X.columns:
            raise KeyError(f"Columns '{self.humidity_col}' and/or '{self.dewpoint_col}' not found in input data.")

        # Create the interaction feature
        X[self.interaction_col] = X[self.humidity_col] * X[self.dewpoint_col]
        return X

# 7. Cloud Sun Interaction Transformer
class CloudSunInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cloud_col='cloud_cover_mean', 
                 sun_col='sunshine_duration', 
                 interaction_col='cloud_sun_interaction'):
        self.cloud_col = cloud_col
        self.sun_col = sun_col
        self.interaction_col = interaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.cloud_col not in X.columns or self.sun_col not in X.columns:
            raise KeyError(f"Columns '{self.cloud_col}' and/or '{self.sun_col}' not found in input data.")

        # Create the interaction feature
        X[self.interaction_col] = X[self.cloud_col] * X[self.sun_col]
        return X

# 8. Radiation Per Sunhour Transformer
class RadiationPerSunhourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, radiation_col='shortwave_radiation_sum', 
                 sun_col='sunshine_duration', 
                 interaction_col='radiation_per_sunhour'):
        self.radiation_col = radiation_col
        self.sun_col = sun_col
        self.interaction_col = interaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.radiation_col not in X.columns or self.sun_col not in X.columns:
            raise KeyError(f"Columns '{self.radiation_col}' and/or '{self.sun_col}' not found in input data.")

        # Avoid division by zero
        X[self.interaction_col] = X[self.radiation_col] / (X[self.sun_col].replace(0, 1e-6))
        return X

# 9. Log1p From List Transformer
class Log1pFromListTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_transform):
        self.columns_to_transform = columns_to_transform
        self.transform_info = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        self.transform_info = {}
        
        for col in self.columns_to_transform:
            if col not in X.columns:
                continue
                
            # Coerce to numeric if needed
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = pd.to_numeric(X[col], errors='coerce')

            col_min = X[col].min(skipna=True)
            if pd.isna(col_min):
                print(f"Skipping {col}: all values NaN")
                continue

            shift = 0.0
            if col_min <= -1.0:
                shift = float(abs(col_min) + 1e-6)
                X[col] = X[col] + shift  # Make values > -1

            # Apply log1p (safe for zeros and values > -1)
            X[col] = np.log1p(X[col])
            self.transform_info[col] = {'shift_added': shift}

        return X

# 9. One Hot Encode Transformer
class OneHotEncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_encode, prefix=None, drop_first=False, all_possibilities=None):
        self.column_to_encode = column_to_encode
        self.prefix = prefix
        self.drop_first = drop_first
        self.columns_ = None
        self.all_possibilities = all_possibilities  # List of all possible categories

    def fit(self, X, y=None):
        if self.column_to_encode not in X.columns:
            raise KeyError(f"Column '{self.column_to_encode}' not found in input data.")
        # Use all_possibilities if provided, else infer from data
        if self.all_possibilities is not None:
            categories = self.all_possibilities
        else:
            categories = X[self.column_to_encode].unique()
        # Create a dummy DataFrame to get all columns
        dummy_df = pd.DataFrame({self.column_to_encode: categories})
        encoded = pd.get_dummies(dummy_df, prefix=self.prefix, drop_first=self.drop_first)
        self.columns_ = encoded.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        if self.column_to_encode not in X.columns:
            raise KeyError(f"Column '{self.column_to_encode}' not found in input data.")
        # Perform one-hot encoding
        encoded = pd.get_dummies(X[self.column_to_encode], prefix=self.prefix, drop_first=self.drop_first)
        # Ensure all expected columns are present (add missing ones with 0s)
        for col in self.columns_:
            if col not in encoded.columns:
                encoded[col] = 0
        # Reorder columns to match training order
        encoded = encoded[self.columns_]
        X = pd.concat([X, encoded], axis=1)
        X.drop(columns=[self.column_to_encode], inplace=True)
        return X

# 11. MinMax Scaler Transformer
class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='precipitation_sum_next_4_days'):
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.numeric_features = None

    def fit(self, X, y=None):
        # Select numeric columns excluding the target column
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.drop(self.target_column, errors='ignore')
        self.scaler.fit(X[self.numeric_features])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        return X

# 12. Dataset Splitter Transformer
class DatasetSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, train_size=0.7, val_size=0.15):
        self.target_column = target_column
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = 1 - train_size - val_size
        self.splits = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure data is sorted by time if 'date' column exists
        if 'date' in X.columns:
            X = X.sort_values(by='date', ascending=True)
            X.drop(columns=['date'], inplace=True, errors='ignore')

        # Define split indices
        train_size = int(len(X) * self.train_size)
        val_size = int(len(X) * self.val_size)

        # Split data
        train = X.iloc[:train_size]
        val = X.iloc[train_size:train_size + val_size]
        test = X.iloc[train_size + val_size:]

        # Separate features and target
        self.splits = {
            'X_train': train.drop(columns=[self.target_column]),
            'y_train': train[self.target_column],
            'X_val': val.drop(columns=[self.target_column]),
            'y_val': val[self.target_column],
            'X_test': test.drop(columns=[self.target_column]),
            'y_test': test[self.target_column]
        }
        return self.splits
print("\n")

print("\n")
print("All transformers for the precipitation fall pipeline defined successfully!")
print("\n")


# Define all the custom transformer classes first
class WindDirectionCompassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_column='wind_direction_10m_dominant', 
                 degree_column='wind_direction_10m_dominant_deg', 
                 compass_column='wind_dir_compass'):
        self.input_column = input_column
        self.degree_column = degree_column
        self.compass_column = compass_column
        self.dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                     'S','SSW','SW','WSW','W','WNW','NW','NNW']

    def deg_to_compass(self, deg):
        """Convert degrees to 16-point compass direction"""
        try:
            deg = float(deg)
        except Exception:
            return np.nan
        idx = int(((deg + 11.25) % 360) / 22.5)
        return self.dirs[idx]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.input_column not in X.columns:
            raise KeyError(f"Column '{self.input_column}' not found in input data.")
        
        X[self.degree_column] = pd.to_numeric(X[self.input_column], errors='coerce')
        X[self.compass_column] = X[self.degree_column].apply(
            lambda x: self.deg_to_compass(x) if not pd.isna(x) else np.nan
        )
        return X

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns_to_drop, errors='ignore')

class CapOutliersFromDict(BaseEstimator, TransformerMixin):
    def __init__(self, bounds_dict):
        self.bounds_dict = bounds_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, bounds in self.bounds_dict.items():
            if col not in X.columns:
                continue
                
            lower_fill = bounds['lower_fill']
            upper_fill = bounds['upper_fill']

            X[col] = pd.to_numeric(X[col], errors='coerce')
            X.loc[X[col] < lower_fill, col] = lower_fill
            X.loc[X[col] > upper_fill, col] = upper_fill
        return X

class HumidityDewPointInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, humidity_col='relative_humidity_2m_mean', dewpoint_col='dew_point_2m_mean', interaction_col='humidity_dewpoint_interaction'):
        self.humidity_col = humidity_col
        self.dewpoint_col = dewpoint_col
        self.interaction_col = interaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.humidity_col not in X.columns or self.dewpoint_col not in X.columns:
            raise KeyError(f"Columns '{self.humidity_col}' and/or '{self.dewpoint_col}' not found in input data.")
        X[self.interaction_col] = X[self.humidity_col] * X[self.dewpoint_col]
        return X

class CloudSunInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cloud_col='cloud_cover_mean', sun_col='sunshine_duration', interaction_col='cloud_sun_interaction'):
        self.cloud_col = cloud_col
        self.sun_col = sun_col
        self.interaction_col = interaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.cloud_col not in X.columns or self.sun_col not in X.columns:
            raise KeyError(f"Columns '{self.cloud_col}' and/or '{self.sun_col}' not found in input data.")
        X[self.interaction_col] = X[self.cloud_col] * X[self.sun_col]
        return X

class RadiationPerSunhourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, radiation_col='shortwave_radiation_sum', sun_col='sunshine_duration', interaction_col='radiation_per_sunhour'):
        self.radiation_col = radiation_col
        self.sun_col = sun_col
        self.interaction_col = interaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.radiation_col not in X.columns or self.sun_col not in X.columns:
            raise KeyError(f"Columns '{self.radiation_col}' and/or '{self.sun_col}' not found in input data.")
        X[self.interaction_col] = X[self.radiation_col] / (X[self.sun_col].replace(0, 1e-6))
        return X

class Log1pFromListTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_transform):
        self.columns_to_transform = columns_to_transform
        self.transform_info = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        self.transform_info = {}
        
        for col in self.columns_to_transform:
            if col not in X.columns:
                continue
                
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = pd.to_numeric(X[col], errors='coerce')

            col_min = X[col].min(skipna=True)
            if pd.isna(col_min):
                continue

            shift = 0.0
            if col_min <= -1.0:
                shift = float(abs(col_min) + 1e-6)
                X[col] = X[col] + shift

            X[col] = np.log1p(X[col])
            self.transform_info[col] = {'shift_added': shift}

        return X

# 9. One Hot Encode Transformer
class OneHotEncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_encode, prefix=None, drop_first=False, all_possibilities=None):
        self.column_to_encode = column_to_encode
        self.prefix = prefix
        self.drop_first = drop_first
        self.columns_ = None
        self.all_possibilities = all_possibilities  # List of all possible categories

    def fit(self, X, y=None):
        if self.column_to_encode not in X.columns:
            raise KeyError(f"Column '{self.column_to_encode}' not found in input data.")
        # Use all_possibilities if provided, else infer from data
        if self.all_possibilities is not None:
            categories = self.all_possibilities
        else:
            categories = X[self.column_to_encode].unique()
        # Create a dummy DataFrame to get all columns
        dummy_df = pd.DataFrame({self.column_to_encode: categories})
        encoded = pd.get_dummies(dummy_df, prefix=self.prefix, drop_first=self.drop_first)
        self.columns_ = encoded.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        if self.column_to_encode not in X.columns:
            raise KeyError(f"Column '{self.column_to_encode}' not found in input data.")
        # Perform one-hot encoding
        encoded = pd.get_dummies(X[self.column_to_encode], prefix=self.prefix, drop_first=self.drop_first)
        # Ensure all expected columns are present (add missing ones with 0s)
        for col in self.columns_:
            if col not in encoded.columns:
                encoded[col] = 0
        # Reorder columns to match training order
        encoded = encoded[self.columns_]
        X = pd.concat([X, encoded], axis=1)
        X.drop(columns=[self.column_to_encode], inplace=True)
        return X

class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='precipitation_sum_next_4_days'):
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.numeric_features = None

    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.drop(self.target_column, errors='ignore')
        if len(self.numeric_features) > 0:
            self.scaler.fit(X[self.numeric_features])
        return self

    def transform(self, X):
        X = X.copy()
        if self.numeric_features is not None and len(self.numeric_features) > 0:
            # Only transform columns that exist in both training and current data
            available_features = [col for col in self.numeric_features if col in X.columns]
            if len(available_features) > 0:
                X[available_features] = self.scaler.transform(X[available_features])
        return X

# Prediction Pipeline Class
class ModelPredictionPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, model_paths, model_names=None):
        self.model_paths = model_paths
        self.model_names = model_names or [f"Model_{i+1}" for i in range(len(model_paths))]
        self.models = {}
        self.preprocessing_pipeline = None
        
    def load_models(self):
        """Load all saved models"""
        if isinstance(self.model_paths, dict):
            for name, path in self.model_paths.items():
                try:
                    self.models[name] = joblib.load(path)
                    print(f"Successfully loaded {name} from {path}")
                except Exception as e:
                    print(f"Error loading {name}: {e}")
        else:
            for i, path in enumerate(self.model_paths):
                try:
                    model_name = self.model_names[i]
                    self.models[model_name] = joblib.load(path)
                    print(f"Successfully loaded {model_name} from {path}")
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")
                    
    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline (without target creation and dataset splitting)"""
        self.preprocessing_pipeline = Pipeline(steps=[
            # Step 0: Drop date column first
            ('drop_date_column', DropColumnsTransformer(columns_to_drop=['date'])),
            
            # Step 1: Wind direction processing
            ('wind_direction_compass', WindDirectionCompassTransformer(
                input_column='wind_direction_10m_dominant',
                degree_column='wind_direction_10m_dominant_deg',
                compass_column='wind_dir_compass'
            )),
            
            # Step 2: Feature selection - Remove data leakage columns
            ('drop_leakage_columns', DropColumnsTransformer(
                columns_to_drop=['weather_code', 'snowfall_sum', 'snowfall_water_equivalent_sum',
                                'precipitation_sum', 'precipitation_hours', 'rain_sum']
            )),
            
            # Step 3: Remove low correlation features
            ('drop_low_correlation', DropColumnsTransformer(
                columns_to_drop=['temperature_2m_max', 'apparent_temperature_max', 'sunrise', 'sunset', 
                                'daylight_duration', 'pressure_msl_mean', 'pressure_msl_max', 'pressure_msl_min', 
                                'surface_pressure_mean', 'surface_pressure_max', 'surface_pressure_min', 
                                'wind_gusts_10m_min', 'wind_speed_10m_min', 'soil_moisture_0_to_100cm_mean', 
                                'soil_moisture_28_to_100cm_mean', 'soil_temperature_0_to_7cm_mean']
            )),
            
            # Step 4: Handle outliers
            ('cap_outliers', CapOutliersFromDict(bounds_dict={
                'temperature_2m_mean': {'lower_fill': 0.0, 'upper_fill': 1.0390099263708124},
                'temperature_2m_min': {'lower_fill': 0.0, 'upper_fill': 1.0870135293646832},
                'apparent_temperature_mean': {'lower_fill': 0.0, 'upper_fill': 1.1140284047672626},
                'apparent_temperature_min': {'lower_fill': 0.0, 'upper_fill': 1.1768906411625464},
                'wind_speed_10m_max': {'lower_fill': 0.132333338294502, 'upper_fill': 1.134174161783695},
                'wind_gusts_10m_max': {'lower_fill': 0.0, 'upper_fill': 1.0000000000000002},
                'wind_direction_10m_dominant': {'lower_fill': 0.0, 'upper_fill': 1.5440955916309815},
                'shortwave_radiation_sum': {'lower_fill': 0.0, 'upper_fill': 1.1560502283105025},
                'et0_fao_evapotranspiration': {'lower_fill': 0.0, 'upper_fill': 1.0},
                'sunshine_duration': {'lower_fill': 0.0, 'upper_fill': 1.270743538679772},
                'cloud_cover_mean': {'lower_fill': 0.0, 'upper_fill': 1.4435416475},
                'cloud_cover_max': {'lower_fill': 0.0, 'upper_fill': 1.6000000000000008},
                'cloud_cover_min': {'lower_fill': 0.0, 'upper_fill': 1.9152894957584379},
                'dew_point_2m_mean': {'lower_fill': 0.0, 'upper_fill': 1.2299532485531657},
                'dew_point_2m_max': {'lower_fill': 0.0, 'upper_fill': 1.063789279112754},
                'dew_point_2m_min': {'lower_fill': 0.0, 'upper_fill': 1.2636629097182288},
                'et0_fao_evapotranspiration_sum': {'lower_fill': 0.0, 'upper_fill': 1.0},
                'relative_humidity_2m_mean': {'lower_fill': 3.3306690738754696e-16, 'upper_fill': 1.1252190464170844},
                'relative_humidity_2m_max': {'lower_fill': 0.0, 'upper_fill': 1.3812069872838006},
                'relative_humidity_2m_min': {'lower_fill': 5.551115123125783e-17, 'upper_fill': 1.0244844611079011},
                'winddirection_10m_dominant': {'lower_fill': 0.0, 'upper_fill': 1.5440955916309815},
                'wind_gusts_10m_mean': {'lower_fill': 0.053925895421582604, 'upper_fill': 1.1449946270239484},
                'wind_speed_10m_mean': {'lower_fill': 0.025425608029460978, 'upper_fill': 1.1589530170126627},
                'wet_bulb_temperature_2m_mean': {'lower_fill': 0.0, 'upper_fill': 1.1959062328556291},
                'wet_bulb_temperature_2m_max': {'lower_fill': 0.0, 'upper_fill': 1.0348707172569274},
                'wet_bulb_temperature_2m_min': {'lower_fill': 0.0, 'upper_fill': 1.2145030202088112},
                'vapour_pressure_deficit_max': {'lower_fill': 0.03462137583318187, 'upper_fill': 1.1007901896473604},
                'soil_moisture_0_to_7cm_mean': {'lower_fill': 0.0, 'upper_fill': 1.0682138384790962},
                'soil_moisture_7_to_28cm_mean': {'lower_fill': 0.0, 'upper_fill': 1.0000000000000002},
                'soil_temperature_0_to_100cm_mean': {'lower_fill': 0.0, 'upper_fill': 1.365240861524692},
                'soil_temperature_7_to_28cm_mean': {'lower_fill': 0.0, 'upper_fill': 1.2335928570691173},
                'soil_temperature_28_to_100cm_mean': {'lower_fill': 0.0, 'upper_fill': 1.3914280692058174},
                'wind_direction_10m_dominant_deg': {'lower_fill': 0.0, 'upper_fill': 1.5440955916309815},
                'temp_humidity_interaction': {'lower_fill': 0.0, 'upper_fill': 1.1503004218330197},
                'cloud_sun_interaction': {'lower_fill': 0.0, 'upper_fill': 0.9289343783125318}
            })),
            
            # Step 5: Feature engineering - Create interaction features
            ('humidity_dewpoint_interaction', HumidityDewPointInteractionTransformer(
                humidity_col='relative_humidity_2m_mean',
                dewpoint_col='dew_point_2m_mean',
                interaction_col='humidity_dewpoint_interaction'
            )),
            ('cloud_sun_interaction', CloudSunInteractionTransformer(
                cloud_col='cloud_cover_mean',
                sun_col='sunshine_duration',
                interaction_col='cloud_sun_interaction'
            )),
            ('radiation_per_sunhour', RadiationPerSunhourTransformer(
                radiation_col='shortwave_radiation_sum',
                sun_col='sunshine_duration',
                interaction_col='radiation_per_sunhour'
            )),
            
            # Step 6: Handle skewness transformation
            ('log1p_transform', Log1pFromListTransformer(columns_to_transform=['radiation_per_sunhour'])),
            
            # Step 7: One-hot encode categorical features
            ('one_hot_encode', OneHotEncodeTransformer(
                column_to_encode='wind_dir_compass',
                prefix='wind_dir',
                drop_first=False,
                all_possibilities=[
            'N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW'])),
            # ...other steps...
            
            # Step 8: Normalize features
            ('minmax_scaler', MinMaxScalerTransformer(target_column='precipitation_sum_next_4_days'))
        ])
    
    def fit(self, X, y=None):
        """Fit the preprocessing pipeline"""
        self.load_models()
        self.create_preprocessing_pipeline()
        # Fit the preprocessing pipeline with training data
        if hasattr(X, 'columns'):
            # Use the provided data to fit the preprocessing pipeline
            sample_data = X.copy()
            # Add dummy target column if it doesn't exist
            if 'precipitation_sum_next_4_days' not in sample_data.columns:
                sample_data['precipitation_sum_next_4_days'] = 0
            self.preprocessing_pipeline.fit(sample_data)
        return self
        
    def preprocess_data(self, X):
        """Apply preprocessing to input data"""
        if self.preprocessing_pipeline is None:
            self.create_preprocessing_pipeline()
        
        X_processed = X.copy()
        
        try:
            X_processed = self.preprocessing_pipeline.transform(X_processed)
            return X_processed
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            
            traceback.print_exc()
            return None
    
    def predict_single_model(self, X, model_name):
        """Make predictions using a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        X_processed = self.preprocess_data(X)
        if X_processed is None:
            return None
            
        predictions = self.models[model_name].predict(X_processed)
        return predictions
    
    def predict_all_models(self, X):
        """Make predictions using all loaded models"""
        X_processed = self.preprocess_data(X)
        if X_processed is None:
            return None
            
        predictions = {}
        for model_name, model in self.models.items():
            try:
                predictions[model_name] = model.predict(X_processed)
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = None
                
        return predictions
    
    def predict_ensemble(self, X, method='mean'):
        """Make ensemble predictions using all models"""
        all_predictions = self.predict_all_models(X)
        
        if not all_predictions or all(pred is None for pred in all_predictions.values()):
            return None
            
        valid_predictions = [pred for pred in all_predictions.values() if pred is not None]
        
        if not valid_predictions:
            return None
            
        predictions_array = np.column_stack(valid_predictions)
        
        if method == 'mean':
            ensemble_pred = np.mean(predictions_array, axis=1)
        elif method == 'median':
            ensemble_pred = np.median(predictions_array, axis=1)
        elif method == 'max':
            ensemble_pred = np.max(predictions_array, axis=1)
        elif method == 'min':
            ensemble_pred = np.min(predictions_array, axis=1)
        else:
            raise ValueError("Method must be one of: 'mean', 'median', 'max', 'min'")
            
        return ensemble_pred

# Initialize the prediction pipeline
model_paths = {
    'CatBoost': r'models/precipitation_fall/catboost_best_model.pkl',
    'GradientBoosting': r'models/precipitation_fall/gbr_best_model.pkl',
    'LightGBM': r'models/precipitation_fall/reg_lgbm_best_model.pkl'
}

# Create and fit the prediction pipeline
predictor = ModelPredictionPipeline(model_paths)

# Load the training data to fit the preprocessing pipeline properly
try:
    # Load the original raw data and use a representative sample for fitting
    raw_data_sample = pd.read_csv(r'data/raw/weather_data.csv').head(500)
    predictor.fit(raw_data_sample)
    print("Prediction pipeline created and fitted successfully!")
    print(f"Loaded models: {list(predictor.models.keys())}")
except Exception as e:
    print(f"Error fitting prediction pipeline: {e}")
    
    traceback.print_exc()

# Simplified prediction example
def simple_prediction_test():
    """Simple test to verify the prediction pipeline works"""
    
    # Load original raw data
    raw_data = pd.read_csv(r'data/raw/weather_data.csv')
    
    # Take a small sample for testing
    test_sample = raw_data.iloc[100:110].copy()  # 10 rows for testing
    print("\n")
    print("=== Testing Rain_fall Pipeline with All Three Models ===")
    print("\n")
    print("Testing prediction pipeline with raw data sample...")
    # print(f"Input data shape: {test_sample.shape}")
    # print(f"Input columns: {list(test_sample.columns)}")
    
    try:
        # Test ensemble prediction
        predictions = predictor.predict_ensemble(test_sample, method='mean')
        
        if predictions is not None:
            print(f"\nSuccess! Got {len(predictions)} predictions")
            print(f"Sample predictions: {predictions}")
            print(f"Prediction range: {np.min(predictions):.2f} to {np.max(predictions):.2f}")
            print("\n")
            print("=== Testing Ensemble Pipeline with All Three Models ===")
            print("\n")
        else:
            print("Failed to get predictions")
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        
        traceback.print_exc()

# Run the simple test
simple_prediction_test()


# Example usage for making predictions - CORRECTED VERSION

# 1. Load new data for prediction and ensure it has the same structure as training data
new_data = pd.read_csv(r'data/raw/weather_data.csv')

# Important: Use the same raw data structure that was used during training
# The prediction pipeline expects the raw data format, not the preprocessed X_test

# 2. Use new_data (or a subset) for predictions instead of X_test
# Let's use a sample from the raw data for demonstration
sample_data = new_data.head(100).copy()  # Use first 100 rows as example

# Make predictions with individual models
try:
    catboost_predictions = predictor.predict_single_model(sample_data, 'CatBoost')
    lgbm_predictions = predictor.predict_single_model(sample_data, 'LightGBM')
    gbr_predictions = predictor.predict_single_model(sample_data, 'GradientBoosting')
    
    # Make predictions with all models
    all_predictions = predictor.predict_all_models(sample_data)
    
    # Make ensemble predictions
    ensemble_mean = predictor.predict_ensemble(sample_data, method='mean')
    ensemble_median = predictor.predict_ensemble(sample_data, method='median')
    
    # Display results
    results_df = pd.DataFrame({
        'CatBoost': catboost_predictions,
        'LightGBM': lgbm_predictions,
        'GradientBoosting': gbr_predictions,
        'Ensemble_Mean': ensemble_mean,
        'Ensemble_Median': ensemble_median
    })
    print("\n")
    print("Prediction Results:")
    print(results_df.head(10))
    
    # Calculate performance metrics for ensemble
    if ensemble_mean is not None:
        print(f"\nEnsemble Predictions Summary:")
        print(f"Mean Prediction: {np.mean(ensemble_mean):.4f}")
        print(f"Std Prediction: {np.std(ensemble_mean):.4f}")
        print(f"Min Prediction: {np.min(ensemble_mean):.4f}")
        print(f"Max Prediction: {np.max(ensemble_mean):.4f}")
    
except Exception as e:
    print(f"Error in prediction pipeline: {e}")
    
    # Alternative approach: Test with the original training data structure
    print("\nTrying with original training data format...")
    
    # Create a sample that matches the original data structure
    # Use the test_data dataframe (original data) for testing
    test_data = pd.read_csv(r'data/raw/weather_data.csv')
    test_sample = test_data.head(50).copy()
    
    try:
        catboost_predictions = predictor.predict_single_model(test_sample, 'CatBoost')
        ensemble_mean = predictor.predict_ensemble(test_sample, method='mean')
        
        if catboost_predictions is not None:
            print("Success! Predictions made:")
            print(f"Sample predictions: {catboost_predictions[:5]}")
            print(f"Ensemble mean predictions: {ensemble_mean[:5] if ensemble_mean is not None else 'None'}")
        else:
            print("Still getting None predictions - there may be an issue with the model files or preprocessing")
            
    except Exception as e2:
        print(f"Error with alternative approach: {e2}")
print("\n" + "="*50)
