from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "model" / "fake_news_detector.joblib"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Fake News Detector API",
    version="1.0.1",
    description="TF-IDF + LinearSVC classifier trained on the Kaggle Fake and Real News Dataset.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    title: str = Field(default="", max_length=5000)
    text: str = Field(..., min_length=10, max_length=100000)

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    decision_score: float
    message: str

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model file not found. Run: python scripts/train_model.py")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: NewsRequest):
    try:
        model = get_model()
        content = f"{payload.title} {payload.text}".strip()
        prediction = int(model.predict([content])[0])
        score = float(model.decision_function([content])[0])
        label = "FAKE" if prediction == 1 else "REAL"
        return PredictionResponse(
            prediction=prediction,
            label=label,
            decision_score=score,
            message=(
                "Classification completed. This model detects dataset-learned "
                "patterns; it is not a factual verification service."
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# Serve index.html, style.css and app.js from the frontend directory.
# API routes above remain available because they are registered first.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
