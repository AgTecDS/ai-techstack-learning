"""
schemas.py
----------
Pydantic models define the exact shape of API request and response data.
FastAPI uses these at runtime to validate inputs and serialize outputs,
and at startup to generate OpenAPI (Swagger) documentation automatically.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Returned by GET /health — confirms the service is up and model is loaded."""
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    device: str = Field(..., example="cpu")


class PredictionResponse(BaseModel):
    """
    Returned by POST /predict.

    - predicted_digit: the argmax class (0-9)
    - confidence: probability of the predicted class
    - probabilities: full softmax distribution over all 10 classes
    """
    predicted_digit: int = Field(..., ge=0, le=9, example=7)
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.9987)
    probabilities: List[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        example=[0.0001, 0.0002, 0.0005, 0.0003, 0.0001, 0.0001, 0.0001, 0.9987, 0.0, 0.0],
    )


class ErrorResponse(BaseModel):
    """Returned when something goes wrong."""
    detail: str = Field(..., example="Model not loaded. Check the server logs.")
    error_code: Optional[str] = Field(None, example="MODEL_NOT_LOADED")
