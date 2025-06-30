import pandas as pd
import numpy as np
import os

class DataPreprocessing:
    def __init__(self, file_path='data/raw/energydata_complete.csv'):
        """
        Initializes the data preprocessing component.
        Args:
            file_path (str): Path to the raw data file.
        """
        self.file_path = file_path

    def load_and_clean_data(self):
        """
        Loads the dataset, performs initial cleaning, and preprocessing.
        - Converts 'date' column to datetime and sets it as the index.
        - Drops irrelevant 'rv1' and 'rv2' columns.
        - Applies a log transformation to the 'Appliances' target variable to handle skewness.
        
        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        try:
            # Construct the absolute path
            # This assumes the script is run from the project root directory
            full_path = os.path.join(os.getcwd(), self.file_path)
            df = pd.read_csv(full_path)
            print(f"Successfully loaded data from {full_path}")
        except FileNotFoundError:
            print(f"Error: The file was not found at {full_path}.")
            print("Please ensure the dataset is in the 'data/raw/' directory and you are running the script from the project's root folder.")
            return None

        # Convert date column and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Drop irrelevant columns
        df_cleaned = df.drop(columns=['rv1', 'rv2'])
        
        # Log transform the target variable
        df_cleaned['Appliances'] = np.log1p(df_cleaned['Appliances'])
        
        print("Data cleaning and initial preprocessing complete.")
        return df_cleaned

