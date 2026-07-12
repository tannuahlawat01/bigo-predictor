"""
main.py

FastAPI endpoint exposing the time complexity predictor over HTTP.
Mirrors the production patterns from the Node/Express project:
  - Input validation (Pydantic, equivalent to express-validator)
  - Structured, consistent response shape (like ApiResponse/ApiError)
  - Centralized error handling
  - CORS configured for the Streamlit frontend
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent))
from model.predict import predict_complexity

app = FastAPI(
    title="Big-O Predictor API",
    description="Predicts time complexity of Python code using AST-based features",
    version="1.0.0",
)

# CORS: allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # default Streamlit port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Request/response schemas (equivalent to input validation middleware) ----

class CodeInput(BaseModel):
    code: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Python source code to analyze",
    )


class PredictionResponse(BaseModel):
    success: bool
    complexity: str | None = None
    confidence: float | None = None
    all_probabilities: dict | None = None
    features: dict | None = None
    error: str | None = None


# ---- Routes ----

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: CodeInput):
    result = predict_complexity(payload.code)

    if result["error"]:
        # Code failed to parse — this is a client error (bad input),
        # not a server error, so 400 not 500
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "success": True,
        "complexity": result["complexity"],
        "confidence": result["confidence"],
        "all_probabilities": result["all_probabilities"],
        "features": result["features"],
        "error": None,
    }


# ---- Centralized error handling for anything unexpected ----

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {
        "success": False,
        "error": "Internal server error",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)