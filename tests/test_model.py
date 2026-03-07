import pytest
import pandas as pd
from src.model import (
    load_model,
    load_encoders,
    prepare_input,
    predict_salary
)


# ─────────────────────────────────────────────
# TEST 1: Model loads without errors
# ─────────────────────────────────────────────
def test_model_loads():
    """Model file should load successfully"""
    model = load_model()
    assert model is not None


# ─────────────────────────────────────────────
# TEST 2: Encoders load correctly
# ─────────────────────────────────────────────
def test_encoders_load():
    """Encoders should load and contain expected keys"""
    encoders = load_encoders()
    
    assert isinstance(encoders, dict)
    assert "EdLevel" in encoders
    assert "Employment" in encoders
    assert "DevType" in encoders
    assert "OrgSize" in encoders


# ─────────────────────────────────────────────
# TEST 3: Model columns are correct shape
# ─────────────────────────────────────────────
def test_model_columns():
    """Model should have the expected number of columns"""
    model = load_model()
    columns = list(model.feature_names_in_)
    
    assert isinstance(columns, list)
    assert len(columns) > 5
    assert "YearsCodePro" in columns
    assert "EdLevel" in columns


# ─────────────────────────────────────────────
# TEST 4: prepare_input returns correct shape
# ─────────────────────────────────────────────
def test_prepare_input_shape():
    """prepare_input should return DataFrame with correct shape"""
    model = load_model()
    expected_cols = len(model.feature_names_in_)
    
    df = prepare_input(
        country="Germany",
        ed_level="Bachelor",
        years_experience=5,
        employment="Employed, full-time",
        dev_type="Developer, back-end",
        org_size="20 to 99 employees"
    )
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert df.shape[1] == expected_cols


# ─────────────────────────────────────────────
# TEST 5: USA salary > India salary
# ─────────────────────────────────────────────
def test_usa_earns_more_than_india():
    """
    Given identical profiles, USA should predict higher salary than India.
    This tests that the model learned real-world patterns!
    """
    usa_salary = predict_salary(
        country="United States of America",
        ed_level="Master",
        years_experience=5,
        employment="Employed, full-time",
        dev_type="Developer, back-end",
        org_size="100 to 499 employees"
    )
    
    india_salary = predict_salary(
        country="India",
        ed_level="Master",
        years_experience=5,
        employment="Employed, full-time",
        dev_type="Developer, back-end",
        org_size="100 to 499 employees"
    )
    
    assert usa_salary > india_salary


# ─────────────────────────────────────────────
# TEST 6: More experience = higher salary
# ─────────────────────────────────────────────
def test_experience_increases_salary():
    """
    10 years experience should predict higher salary than 1 year.
    Tests that the model learned the experience→salary relationship!
    """
    junior = predict_salary(
        country="Germany",
        ed_level="Bachelor",
        years_experience=1,
        employment="Employed, full-time",
        dev_type="Developer, back-end",
        org_size="100 to 499 employees"
    )
    
    senior = predict_salary(
        country="Germany",
        ed_level="Bachelor",
        years_experience=10,
        employment="Employed, full-time",
        dev_type="Developer, back-end",
        org_size="100 to 499 employees"
    )
    
    assert senior > junior