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