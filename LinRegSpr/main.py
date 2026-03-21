"""
FastAPI — Hooke's Law TensorFlow Web App
훅의 법칙을 TensorFlow 선형회귀로 학습·예측하는 FastAPI 서버
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from model import spring_model, OUTPUT_DIR

# ─── 앱 초기화 ────────────────────────────────────────────────────────
app = FastAPI(
    title="Hooke's Law — TensorFlow Demo",
    description="훅의 법칙 선형회귀 학습 및 예측 웹앱",
    version="1.0.0"
)

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_PATH   = Path(OUTPUT_DIR)

# Static files (PNG output)
app.mount("/output", StaticFiles(directory=str(OUTPUT_PATH)), name="output")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─── 요청/응답 스키마 ────────────────────────────────────────────────
class TrainResponse(BaseModel):
    status: str
    epochs: int
    final_loss: float
    final_mae: float
    learned_k: float
    learned_b: float
    true_k: float
    true_b: float
    loss_plot: str
    regression_plot: str
    training_time_sec: float

class PredictRequest(BaseModel):
    mass: float = Field(..., ge=0, le=100, description="질량 (kg), 0~100")

class PredictResponse(BaseModel):
    mass: float
    predicted_length: float
    prediction_plot: str


# ─── 라우터 ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 웹 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train", response_model=TrainResponse)
async def train(epochs: int = 500):
    """TensorFlow 모델 학습"""
    try:
        # 모델 재초기화 (반복 학습 지원)
        spring_model.build()

        t0 = time.time()
        result = spring_model.train(epochs=epochs)
        elapsed = round(time.time() - t0, 2)

        # PNG 저장
        spring_model.save_loss_plot()
        spring_model.save_regression_plot()

        return TrainResponse(
            status="success",
            training_time_sec=elapsed,
            loss_plot="/output/loss_curve.png",
            regression_plot="/output/spring_regression.png",
            **result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    """질량 → 용수철 길이 예측"""
    if not spring_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="모델이 학습되지 않았습니다. /train 먼저 호출하세요."
        )
    try:
        pred = spring_model.predict(body.mass)
        spring_model.save_prediction_plot(body.mass, pred)

        return PredictResponse(
            mass=body.mass,
            predicted_length=round(pred, 4),
            prediction_plot="/output/prediction_result.png"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """모델 상태 확인"""
    return {
        "is_trained": spring_model.is_trained,
        "model_ready": spring_model.model is not None,
        "output_dir": str(OUTPUT_PATH),
        "output_files": [f.name for f in OUTPUT_PATH.iterdir() if f.suffix == '.png']
        if OUTPUT_PATH.exists() else []
    }


# ─── 엔트리포인트 ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
