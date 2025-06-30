# test_preprocessing.py
from src.appliance_energy_predictor.components.data_preprocessing import DataPreprocessing

def test_data_preprocessing():
    print("--- Testing DataPreprocessing Component ---")
    
    # 1. Initialize the component
    preprocessor = DataPreprocessing()
    
    # 2. Run the method you want to test
    df_cleaned = preprocessor.load_and_clean_data()
    
    # 3. Check the result
    if df_cleaned is not None:
        print("\nTest successful! The component ran without errors.")
        print("Here's a sample of the processed data:")
        print(df_cleaned.head())
        print("\nDataFrame Info:")
        df_cleaned.info()
    else:
        print("\nTest failed. The component returned None.")

if __name__ == "__main__":
    test_data_preprocessing()


# test_feature_engineering.py
from src.appliance_energy_predictor.components.data_preprocessing import DataPreprocessing
from src.appliance_energy_predictor.components.feature_engineering import FeatureEngineering

def test_feature_engineering():
    print("--- Testing FeatureEngineering Component ---")
    
    # 1. First, we need the output from the previous step (data preprocessing)
    preprocessor = DataPreprocessing()
    df_cleaned = preprocessor.load_and_clean_data()
    
    if df_cleaned is None:
        print("Data preprocessing failed, cannot proceed with feature engineering test.")
        return

    # 2. Initialize and run the feature engineering component
    feature_engineer = FeatureEngineering(df_cleaned)
    df_featured = feature_engineer.create_features()

    # 3. Check the result
    if df_featured is not None:
        print("\nTest successful! Feature engineering ran without errors.")
        
        # Check if new columns were added
        new_cols = ['hour', 'day_of_week', 'month', 'appliances_lag_1h', 'appliances_lag_2h', 'T_out_rolling_mean_1h', 'appliances_rolling_mean_1h']
        all_cols_present = all(col in df_featured.columns for col in new_cols)
        
        if all_cols_present:
            print("All new feature columns were successfully added.")
        else:
            print("Error: Some feature columns are missing.")
            
        # Check if NaN values were dropped
        if df_featured.isnull().sum().sum() == 0:
            print("No NaN values found in the final dataframe.")
        else:
            print("Error: NaN values still exist in the dataframe.")
            
        print("\nHere's a sample of the final featured data:")
        print(df_featured.head())
    else:
        print("\nTest failed. The component returned None.")


if __name__ == "__main__":
    test_feature_engineering()


# test_model_dev.py
from src.appliance_energy_predictor.components.model_dev import ModelDevelopment
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

def test_model_development():
    print("--- Testing ModelDevelopment Component ---")
    
    model_developer = ModelDevelopment()
    
    lr_model = model_developer.get_linear_regression_model()
    print(f"\nSuccessfully created Linear Regression model: {isinstance(lr_model, LinearRegression)}")
    print(lr_model)
    
    rf_model = model_developer.get_random_forest_model()
    print(f"\nSuccessfully created Random Forest model: {isinstance(rf_model, RandomForestRegressor)}")
    print(rf_model)
    
    dummy_input_shape = (6, 33) 
    lstm_model = model_developer.get_lstm_model(input_shape=dummy_input_shape)
    print(f"\nSuccessfully created LSTM model: {isinstance(lstm_model, tf.keras.Model)}")
    
    first_layer_is_lstm = isinstance(lstm_model.layers[0], tf.keras.layers.LSTM)
    print(f"First layer is an LSTM layer: {first_layer_is_lstm}")


if __name__ == "__main__":
    test_model_development()

