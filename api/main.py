"""
main.py

FastAPI endpoint exposing the time complexity predictor over HTTP.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent))

# Ensure a trained model exists before importing predict.py (which loads
# the model lazily on first request). On first deploy, or any environment
# without a pre-trained model committed, train one from the committed
# feature_matrix.csv rather than failing at request time.

from model.predict import predict_complexity

app = FastAPI(
    title="Big-O Predictor API",
    description="Predicts time complexity of Python code using AST-based features",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "https://*.streamlit.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    explanation: str | None = None
    error: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: CodeInput):
    result = predict_complexity(payload.code)

    if result["error"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "success": True,
        "complexity": result["complexity"],
        "confidence": result["confidence"],
        "all_probabilities": result["all_probabilities"],
        "features": result["features"],
        "explanation": result["explanation"],
        "error": None,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)