import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier # Changed from LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import sys
import os

# Add the parent directory of src to the system path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import get_clean_data # Import from new utils file

def create_model(data):
    """
    Preprocesses the data, trains a RandomForestClassifier model,
    and evaluates its performance.

    Args:
        data (pd.DataFrame): The cleaned dataset.

    Returns:
        tuple: A tuple containing the trained model and the scaler object.
    """
    print("Starting model creation and training...")

    # Separate features (X) and target (y)
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale the data using StandardScaler
    # This step is crucial for many ML models to perform optimally
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaled using StandardScaler.")

    # Split the scaled data into training and testing sets
    # test_size=0.2 means 20% of data for testing, random_state ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"Data split: Training samples = {X_train.shape[0]}, Testing samples = {X_test.shape[0]}")

    # Initialize the RandomForestClassifier
    # RandomForest is chosen for its robustness and good performance on tabular data
    model = RandomForestClassifier(random_state=42)

    # Define a smaller, more focused grid for hyperparameter tuning for speed
    # In a real project, you'd explore a wider range.
    param_grid = {
        'n_estimators': [100, 200],  # Number of trees in the forest
        'max_depth': [10, 20],       # Maximum depth of the tree
        'min_samples_split': [2, 5], # Minimum number of samples required to split an internal node
    }
    print("Starting GridSearchCV for hyperparameter tuning...")

    # Use GridSearchCV for exhaustive search over specified parameter values
    # cv=5 means 5-fold cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best model found by GridSearchCV
    best_model = grid_search.best_estimator_
    print(f"Best model parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Test the best model on the unseen test set
    y_pred = best_model.predict(X_test)

    # Evaluate model performance
    print("\n--- Model Evaluation on Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Model training and evaluation complete.")

    return best_model, scaler


def main():
    """
    Main function to orchestrate data loading, model training,
    and saving the trained model and scaler.
    """
    print("Running model training script...")
    data = get_clean_data()
    print("Data loaded and cleaned.")

    model, scaler = create_model(data)

    # Ensure the 'model' directory exists
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # Save the trained model
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    # Save the fitted scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")


if __name__ == '__main__':
    main()