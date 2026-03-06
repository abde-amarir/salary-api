import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Build the path to data/processed/ regardless of where
# this file is called from
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'data' / 'processed' / 'best_model.pkl'
ENCODERS_PATH = BASE_DIR / 'data' / 'processed' / 'label_encoders.pkl'

def load_model():
    """
    Load the trained Gradient Boosting model from disk.
    Returns the model object ready to make predictions.
    """
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def load_encoders():
    """
    Load the saved LabelEncoders for EdLevel, Employment,
    DevType, and OrgSize.
    Returns a dictionary of {column_name: LabelEncoder}
    """
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    return encoders

def get_country_columns():
    """
    Get the EXACT columns the model was trained on.
    Reads directly from the model — no hardcoding, no guessing!
    """
    model = load_model()
    return list(model.feature_names_in_)

def prepare_input(country, ed_level, years_experience,
                  employment, dev_type, org_size):
    """
    Takes raw user inputs and transforms them into the exact
    format the model expects — column names and order included!
    """
    encoders = load_encoders()
    model_columns = get_country_columns()

    # Step 1: Start with ALL zeros for ALL columns
    data = {col: 0 for col in model_columns}

    # Step 2: Label encode categorical columns
    data['EdLevel']    = encoders['EdLevel'].transform([ed_level])[0]
    data['Employment'] = encoders['Employment'].transform([employment])[0]
    data['DevType']    = encoders['DevType'].transform([dev_type])[0]
    data['OrgSize']    = encoders['OrgSize'].transform([org_size])[0]
    
    # Step 3: Years of experience
    data['YearsCodePro'] = float(years_experience)

    # Step 4: One-Hot Encode country
    country_col = f'Country_{country}'
    if country_col in data:
        data[country_col] = 1
    else:
        # Country not in training data → treat as Other
        data['Country_Other'] = 1

    # Step 5: Build DataFrame in EXACT model column order
    df = pd.DataFrame([data], columns=model_columns)
    return df

def predict_salary(country, ed_level, years_experience,
                   employment, dev_type, org_size):
    """
    The main function the app will call.
    Takes user inputs → returns predicted salary in USD.
    """
    model = load_model()
    input_df = prepare_input(
        country, ed_level, years_experience,
        employment, dev_type, org_size
    )
    prediciton = model.predict(input_df)[0]
    return prediciton