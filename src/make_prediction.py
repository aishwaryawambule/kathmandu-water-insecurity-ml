"""
Make predictions for water insecurity in new locations.
Project: Predicting Household Water Insecurity Due to Urban Groundwater Depletion
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

def get_nearest_aquifer_info(lat, lon, aquifer_df):
    """Find the nearest aquifer well and return its properties."""
    # Calculate Euclidean distance (approximation is fine for this scale)
    distances = np.sqrt((aquifer_df['lat'] - lat)**2 + (aquifer_df['lon'] - lon)**2)
    nearest_idx = distances.idxmin()
    nearest_well = aquifer_df.iloc[nearest_idx]
    
    return nearest_well

def predict_water_insecurity(lat, lon, district, season, household_size, 
                           uses_piped, uses_well, uses_tanker):
    """
    Predict water insecurity index for a specific location and household configuration.
    """
    # Load resources
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(script_dir, '../outputs')
        data_dir = os.path.join(script_dir, '../data/processed')
        
        model_path = os.path.join(outputs_dir, 'water_insecurity_model.joblib')
        features_path = os.path.join(outputs_dir, 'model_features.joblib')
        aquifer_path = os.path.join(data_dir, 'kathmandu_aquifer_clustered.csv')
        
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        aquifer_df = pd.read_csv(aquifer_path)
    except FileNotFoundError as e:
        print(f"Error loading resources: {e}")
        print("Please run the training pipeline first (python main.py)")
        return None

    # Get aquifer info from nearest well
    nearest_well = get_nearest_aquifer_info(lat, lon, aquifer_df)
    
    # Prepare input data
    input_data = {
        'household_size': household_size,
        'uses_piped_water': uses_piped,
        'uses_groundwater_well': uses_well,
        'uses_tanker': uses_tanker,
        'aquifer_cluster': nearest_well['aquifer_cluster'],
        'water_table_depth_m': nearest_well['water_table_depth_m'],
        'depletion_rate_m_per_year': nearest_well['depletion_rate_m_per_year'],
        'tds_mg_per_L': nearest_well['tds_mg_per_L'],
        'nitrate_mg_per_L': nearest_well['nitrate_mg_per_L']
    }
    
    # Create DataFrame with all feature columns initialized to 0.0 (float)
    # This ensures we have the same structure as the training data, including all one-hot encoded columns.
    df_input = pd.DataFrame(0.0, index=[0], columns=features)
    
    # Fill numerical values
    for col, val in input_data.items():
        if col in df_input.columns:
            df_input.loc[0, col] = val
            
    # Handle One-Hot Encoded columns
    # We need to manually set the correct binary column for categorical variables (season, district).
    # Season
    season_col = f"season_{season}"
    if season_col in df_input.columns:
        df_input.loc[0, season_col] = 1
        
    # District
    district_col = f"district_{district}"
    if district_col in df_input.columns:
        df_input.loc[0, district_col] = 1
        
    # Make prediction
    # Returns the predicted class (0, 1, 2)
    prediction_idx = model.predict(df_input)[0]
    
    # Decode label
    # Load label encoder if available, otherwise map manually
    try:
        le = joblib.load(os.path.join(outputs_dir, 'label_encoder.joblib'))
        prediction_class = le.inverse_transform([prediction_idx])[0]
    except:
        # Fallback if LE not found
        mapping = {0: 'High', 1: 'Low', 2: 'Moderate'} # Note: Check alphabetical order of LE usually
        prediction_class = mapping.get(prediction_idx, "Unknown")
    
    return prediction_class, nearest_well

def main():
    print("=== Water Insecurity Prediction Tool ===")
    
    # Define test cases using actual aquifer locations from the generated data
    # These coordinates are near specific aquifers with different stress levels
    test_cases = [
        {
            "label": "Case 1: Secure Household (Low Risk)",
            # Near AQ0077: Low depletion (0.31 m/yr), shallow depth (6.49m)
            "lat": 27.75, "lon": 85.24, "district": "Bhaktapur", "season": "wet",
            "household_size": 4, "uses_piped": 1, "uses_well": 1, "uses_tanker": 0
        },
        {
            "label": "Case 2: Vulnerable Household (Moderate Risk)",
            # Near AQ0003: Moderate depletion (1.71 m/yr), medium depth (24.94m)
            "lat": 27.75, "lon": 85.28, "district": "Bhaktapur", "season": "dry",
            "household_size": 6, "uses_piped": 0, "uses_well": 0, "uses_tanker": 1
        },
        {
            "label": "Case 3: Critical Zone (High Risk)",
            # Near AQ0073: CRITICAL - Very high depletion (3.15 m/yr), deep water table (51.05m)
            "lat": 27.60, "lon": 85.30, "district": "Kathmandu", "season": "dry",
            "household_size": 8, "uses_piped": 0, "uses_well": 0, "uses_tanker": 1
        }
    ]

    for case in test_cases:
        print(f"\n{'='*40}")
        print(f"Testing {case['label']}")
        print(f"{'='*40}")
        
        result = predict_water_insecurity(
            case["lat"], case["lon"], case["district"], case["season"],
            case["household_size"], case["uses_piped"], case["uses_well"], case["uses_tanker"]
        )
        
        if result:
            prediction_class, nearest_well = result
            
            # Color code output
            reset = "\033[0m"
            if prediction_class == 'High': 
                color = "\033[91m" # Red
            elif prediction_class == 'Moderate':
                color = "\033[93m" # Yellow
            elif prediction_class == 'Low': 
                color = "\033[92m" # Green
            else:
                color = reset
            
            print(f"Prediction: {color}{prediction_class} Risk{reset}")
            print(f"Aquifer Context: Depth {nearest_well['water_table_depth_m']}m, Depletion {nearest_well['depletion_rate_m_per_year']}m/yr")

if __name__ == "__main__":
    main()
