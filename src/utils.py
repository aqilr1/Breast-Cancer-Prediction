import pandas as pd
import os  # Import the os module


def get_clean_data():
    """
    Loads and cleans the Breast Cancer Wisconsin (Diagnostic) dataset.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Get the directory of the current script (utils.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the data.csv file
    # We need to go up one level from 'src' to 'Project_Root', then into 'data'
    data_file_path = os.path.join(script_dir, '..', 'data', 'data.csv')

    print(f"Attempting to load data from: {data_file_path}")  # Debugging line

    try:
        data = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file_path}. Please ensure it exists.")
        # Re-raise the exception or handle it gracefully
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        raise

    # Drop irrelevant columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Map diagnosis to numerical values: Malignant (M) = 1, Benign (B) = 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data