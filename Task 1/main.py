

# --- 1. Import Necessary Libraries ---
print("Step 1: Importing libraries...")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import warnings
# Colab-specific imports for file handling
from google.colab import files
import io

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')
print("Libraries imported successfully.\n")


# --- 2. Define Core Functions ---

def upload_and_load_data():
    """
    Handles file uploading in Google Colab and loads the data into a DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the uploaded data, or None if failed.
    """
    print("Step 2: Upload your 'train.csv' file")
    try:
        uploaded = files.upload()
        # Get the name of the uploaded file
        filename = next(iter(uploaded))
        print(f"\nSuccessfully uploaded '{filename}'")
        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
        return df
    except (StopIteration, FileNotFoundError):
        print("\n⚠️ File upload cancelled or failed. Please run the cell again to upload.")
        return None
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return None


def prepare_data(df):
    """
    Selects relevant features, performs cleaning and feature engineering.

    Args:
        df (pandas.DataFrame): The raw DataFrame from the uploaded file.

    Returns:
        pandas.DataFrame: A cleaned DataFrame with selected features.
    """
    print("\nStep 3: Preparing and cleaning data...")
    # Define the features we need and their new, friendly names
    feature_map = {
        'GrLivArea': 'SquareFootage',
        'BedroomAbvGr': 'Bedrooms',
        'FullBath': 'FullBathrooms',
        'HalfBath': 'HalfBathrooms',
        'SalePrice': 'Price'
    }

    # Check if required columns exist
    required_cols = list(feature_map.keys())
    if not all(col in df.columns for col in required_cols):
        print("Error: The uploaded CSV is missing one or more required columns.")
        print(f"Required columns: {required_cols}")
        return None

    # Select only the columns we need
    df_selected = df[required_cols].copy()

    # Rename columns for clarity
    df_selected.rename(columns=feature_map, inplace=True)

    # Feature Engineering: Create a single 'Bathrooms' feature
    # We'll count a half-bath as 0.5
    df_selected['TotalBathrooms'] = df_selected['FullBathrooms'] + 0.5 * df_selected['HalfBathrooms']

    # Drop the original bathroom columns
    df_final = df_selected[['SquareFootage', 'Bedrooms', 'TotalBathrooms', 'Price']]

    # Handle any missing values by filling with the median
    for col in df_final.columns:
        if df_final[col].isnull().any():
            median_val = df_final[col].median()
            df_final[col].fillna(median_val, inplace=True)
            print(f"  - Filled missing values in '{col}' with median value ({median_val}).")

    print("Data preparation complete.\n")
    return df_final

def train_regression_model(X, y):
    """
    Splits the data, trains a linear regression model, and returns the trained model
    and the split datasets.

    Args:
        X (pandas.DataFrame): The feature data.
        y (pandas.Series): The target data.

    Returns:
        tuple: Contains the trained model, X_train, X_test, y_train, y_test.
    """
    print("Step 4: Training the Linear Regression Model...")
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  - Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model training complete.\n")
    return model, X_train, X_test, y_train, y_test

def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluates the model using the test set and prints performance metrics.

    Args:
        model (LinearRegression): The trained model.
        X_test (pandas.DataFrame): The test features.
        y_test (pandas.Series): The actual test target values.

    Returns:
        dict: A dictionary containing the performance metrics.
    """
    print("Step 5: Evaluating Model Performance...")
    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Print the results
    print(f"  - R-squared (R²): {r2:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"  - Mean Absolute Error (MAE): ${mae:,.2f}")
    print("\nInterpretation:")
    print(f"  - The model explains {r2:.2%} of the variance in house prices.")
    print(f"  - On average, the model's predictions are off by ${mae:,.2f}.")
    print("Evaluation complete.\n")

    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'predictions': y_pred}

def display_visualizations(df, y_test, y_pred):
    """
    Creates and displays interactive visualizations for the model results.

    Args:
        df (pandas.DataFrame): The full prepared dataset.
        y_test (pandas.Series): The actual test target values.
        y_pred (numpy.array): The predicted target values.
    """
    print("Step 6: Generating Interactive Visualizations...")

    # 1. Actual vs. Predicted Prices
    fig1 = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'},
        title='Actual vs. Predicted House Prices'
    )
    fig1.add_shape(
        type='line', x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color='red', dash='dash')
    )
    fig1.update_layout(template="plotly_white")
    fig1.show()

    # 2. Residuals Plot
    residuals = y_test - y_pred
    fig2 = px.scatter(
        x=y_pred, y=residuals,
        labels={'x': 'Predicted Price ($)', 'y': 'Residuals ($)'},
        title='Residuals vs. Predicted Price'
    )
    fig2.add_hline(y=0, line_color='red', line_dash='dash')
    fig2.update_layout(yaxis_title="Residuals (Actual - Predicted) ($)", template="plotly_white")
    fig2.show()

    # 3. 3D Scatter plot of features vs. price
    fig3 = px.scatter_3d(
        df, x='SquareFootage', y='TotalBathrooms', z='Price',
        color='Bedrooms', title='3D View: Price vs. Features',
        labels={
            'SquareFootage': 'Square Footage',
            'TotalBathrooms': 'Number of Bathrooms',
            'Price': 'House Price ($)'
        }
    )
    fig3.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    fig3.show()
    print("Visualizations generated. Please check the plot outputs.\n")

def interactive_prediction(model, features):
    """
    Allows the user to input house features and get a price prediction.

    Args:
        model (LinearRegression): The trained model.
        features (list): List of feature names.
    """
    print("Step 7: Interactive House Price Prediction")
    print("-----------------------------------------")
    try:
        sqft = float(input("Enter the Square Footage (e.g., 1500): "))
        bedrooms = int(input("Enter the number of Bedrooms (e.g., 3): "))
        bathrooms = float(input("Enter the number of Bathrooms (e.g., 2.5): "))

        # Create a DataFrame for the input
        input_data = pd.DataFrame([[sqft, bedrooms, bathrooms]], columns=features)

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        print("\n---")
        print(f"🏠 Predicted Price for your house: ${predicted_price:,.2f}")
        print("---\n")

    except ValueError:
        print("\n⚠️ Invalid input. Please enter numerical values for all features.")
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")


# --- 3. Main Execution ---
if __name__ == "__main__":
    # Execute the pipeline
    raw_data = upload_and_load_data()

    if raw_data is not None:
        house_data = prepare_data(raw_data)

        if house_data is not None:
            # Define features (X) and target (y)
            FEATURES = ['SquareFootage', 'Bedrooms', 'TotalBathrooms']
            TARGET = 'Price'

            X = house_data[FEATURES]
            y = house_data[TARGET]

            # Train the model
            trained_model, X_train, X_test, y_train, y_test = train_regression_model(X, y)

            # Evaluate the model
            evaluation_results = evaluate_model_performance(trained_model, X_test, y_test)

            # Display visualizations
            display_visualizations(house_data, y_test, evaluation_results['predictions'])

            # Allow for interactive prediction
            interactive_prediction(trained_model, FEATURES)

            print("✅ Analysis complete!")