"""
Utility functions for the Email Classification and PII Masking application.

This module provides common helper functions that can be used across different
parts of the project, such as data loading, preprocessing, or other shared tasks.
It aims to promote code reusability and organization.
"""
import pandas as pd
from typing import Union

def load_data(file_path: str) -> Union[pd.DataFrame, None]:
    """
    Loads data from a specified CSV file into a pandas DataFrame.

    Args:
        file_path (str): The absolute or relative path to the CSV file.

    Returns:
        Union[pd.DataFrame, None]: A pandas DataFrame containing the loaded data if successful,
                             with 'email' and 'type' columns validated.
                             Returns None if any error occurs during loading or validation
                             (e.g., file not found, empty file, missing required columns).

    Raises:
        Prints an error message to the console if loading fails or if the
        required columns ('email', 'type') are not found in the CSV.
    """
    try:
        df = pd.read_csv(file_path)
        # Basic validation: Check for expected columns 'email' and 'type'
        # Basic validation: Ensure essential columns are present.
        # These columns are critical for training the email classifier and processing emails.
        if 'email' not in df.columns or 'type' not in df.columns:
            print(f"Error: CSV file at {file_path} must contain 'email' and 'type' columns.")
            return None
        print(f"Successfully loaded data from {file_path}. DataFrame shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The data file was not found at the specified path: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The data file at {file_path} is empty and cannot be processed.")
        return None
    except Exception as e:  # Catching other potential pandas or general exceptions during file loading.
        print(f"An unexpected error occurred while loading data from {file_path}: {e}")
        return None

if __name__ == "__main__":
    # This block serves as an example of how to use the functions in this module.
    # It will only execute when this script is run directly (e.g., `python utils.py`)
    # and not when `utils.py` is imported by another module.

    # --- Example: Loading email data --- #
    # Ensure the CSV file 'combined_emails_with_natural_pii.csv' exists in the project's
    # root directory or update DATASET_PATH to the correct location for this example to run.
    # This dataset is assumed to be for demonstration or initial model training preparation.
    DATASET_PATH = 'combined_emails_with_natural_pii.csv'
    email_data = load_data(DATASET_PATH)
    if email_data is not None:
        print(f"Successfully loaded {len(email_data)} emails for example usage.")
        print("First 5 rows:")
        print(email_data.head())
        print("\nEmail categories distribution:")
        print(email_data['type'].value_counts())
