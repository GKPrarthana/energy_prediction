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


