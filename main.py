from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from src.model import predict_salary

# ─────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────
app = FastAPI(
    title="Developer Salary Predictor API",
    description="""
    Predicts developer salaries based on real data from
    42,000+ Stack Overflow survey responses (2023).

    Salaries are in USD and do not account for
    Purchasing Power Parity (PPP).    
    """,
    version="1.0.0"
)

# ─────────────────────────────────────────────
# REQUEST & RESPONSE MODELS
# ─────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """
    This defines EXACTLY what the API expects to receive.
    FastAPI automatically validates every incoming request
    against this model — wrong types or missing fields
    return a clean error automatically!
    """
    country: str = Field(description="Country where the developer works")
    ed_level: str = Field(description="Highest education level")
    years_experience: float = Field(
        ge=0,      # greater than or equal to 0
        le=50,     # less than or equal to 50
        description="Years of professional coding experience"
    )
    employment: str = Field(description="Employment type")
    dev_type: str = Field(description="Type of developer role")
    org_size: str = Field(description="Size of the organization")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "country": "United States of America",
                    "ed_level": "Master",
                    "years_experience": 8,
                    "employment": "Employed, full-time",
                    "dev_type": "Developer, back-end",
                    "org_size": "1,000 to 4,999 employees"
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    """
    This defines EXACTLY what the API always returns.
    Every response is guaranteed to have these fields —
    no surprises for whoever calls our API!
    """
    predicted_salary_usd: float
    lower_bound_usd: float
    upper_bound_usd: float
    model_mae_usd: float
    currency: str
    disclaimer: str

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    """
    Health check endpoint.
    Anyone can call this to verify the API is alive!
    """
    return {
        "status": "🟢 API is running!",
        "name": "Developer Salary Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.get("/health")
def health_check():
    """
    Detailed health check — used by Docker and
    deployment platforms to monitor the service!
    """
    return {
        "status": "healthy",
        "model": "GradientBoostingRegressor",
        "data_source": "Stack Overflow Survey 2023",
        "training_samples": 42723,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Main prediction endpoint.

    Send a POST request with developer info →
    get back a predicted salary in USD!
    """
    MODEL_MAE = 24659.0

    try:
        salary = predict_salary(
            country=request.country,
            ed_level=request.ed_level,
            years_experience=request.years_experience,
            employment=request.employment,
            dev_type=request.dev_type,
            org_size=request.org_size
        )

        return PredictionResponse(
            predicted_salary_usd=round(salary, 2),
            lower_bound_usd=round(max(0, salary - MODEL_MAE), 2),
            upper_bound_usd=round(salary + MODEL_MAE, 2),
            model_mae_usd=MODEL_MAE,
            currency="USD",
            disclaimer=(
                "Salary in USD. Does not account for PPP. "
                "Employer location may differ from developer location."
            )
        ) 
    
    except ValueError as e:
        # Return a clean 422 error if input values are invalid
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input value: {str(e)}"
        )
    except Exception as e:
        # Return a clean 500 error for unexpected problems
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
# ─────────────────────────────────────────────
# RUN THE SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
    )