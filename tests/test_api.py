import pytest
from fastapi.testclient import TestClient
from main import app

# TestClient creates a fake version of your API
# No real server needed — runs entirely in memory!
client = TestClient(app)


# ─────────────────────────────────────────────
# TEST 1: Root endpoint
# ─────────────────────────────────────────────
def test_root_endpoint():
    """API root should return 200 and confirm it's running"""
    response = client.get("/")
    
    assert response.status_code == 200
    assert "status" in response.json()
    assert "running" in response.json()["status"].lower()


# ─────────────────────────────────────────────
# TEST 2: Health check endpoint
# ─────────────────────────────────────────────
def test_health_endpoint():
    """Health check should return model info"""
    response = client.get("/health")
    data = response.json()
    
    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert data["model"] == "GradientBoostingRegressor"
    assert data["training_samples"] == 42723


# ─────────────────────────────────────────────
# TEST 3: Valid prediction returns a number
# ─────────────────────────────────────────────
def test_predict_valid_input():
    """Valid input should return a positive salary prediction"""
    response = client.post("/predict", json={
        "country": "United States of America",
        "ed_level": "Master",
        "years_experience": 8.0,
        "employment": "Employed, full-time",
        "dev_type": "Developer, back-end",
        "org_size": "1,000 to 4,999 employees"
    })
    data = response.json()
    
    assert response.status_code == 200
    assert "predicted_salary_usd" in data
    assert data["predicted_salary_usd"] > 0
    assert data["currency"] == "USD"
    assert data["lower_bound_usd"] < data["predicted_salary_usd"]
    assert data["upper_bound_usd"] > data["predicted_salary_usd"]


# ─────────────────────────────────────────────
# TEST 4: Missing field returns 422
# ─────────────────────────────────────────────
def test_predict_missing_field():
    """Missing required field should return 422 Validation Error"""
    response = client.post("/predict", json={
        "country": "Germany",
        # ed_level is missing!
        "years_experience": 5.0,
        "employment": "Employed, full-time",
        "dev_type": "Developer, back-end",
        "org_size": "20 to 99 employees"
    })
    
    assert response.status_code == 422


# ─────────────────────────────────────────────
# TEST 5: Invalid years_experience rejected
# ─────────────────────────────────────────────
def test_predict_invalid_years():
    """years_experience above 50 should be rejected"""
    response = client.post("/predict", json={
        "country": "Germany",
        "ed_level": "Bachelor",
        "years_experience": 999,  # invalid! max is 50
        "employment": "Employed, full-time",
        "dev_type": "Developer, back-end",
        "org_size": "20 to 99 employees"
    })
    
    assert response.status_code == 422


# ─────────────────────────────────────────────
# TEST 6: Unknown country falls back to "Other"
# ─────────────────────────────────────────────
def test_predict_unknown_country():
    """Unknown country should still return a prediction (falls back to Other)"""
    response = client.post("/predict", json={
        "country": "Mars",  # not in our dataset!
        "ed_level": "Bachelor",
        "years_experience": 3.0,
        "employment": "Employed, full-time",
        "dev_type": "Developer, full-stack",
        "org_size": "20 to 99 employees"
    })
    
    # Should still work — model.py handles unknown countries!
    assert response.status_code == 200
    assert "predicted_salary_usd" in response.json()


# ─────────────────────────────────────────────
# TEST 7: Salary is within realistic bounds
# ─────────────────────────────────────────────
def test_predict_realistic_salary():
    """Predicted salary should be within the training data range"""
    response = client.post("/predict", json={
        "country": "United States of America",
        "ed_level": "PhD",
        "years_experience": 15.0,
        "employment": "Employed, full-time",
        "dev_type": "Data scientist or machine learning specialist",
        "org_size": "10,000 or more employees"
    })
    data = response.json()
    
    assert response.status_code == 200
    # Salary should be between $10k and $250k
    # (our training data range)
    assert 10_000 < data["predicted_salary_usd"] < 250_000