"""
FastAPI — K-Means Clustering Educational Web App
numpy/matplotlib만으로 구현한 비지도 학습 시각화 서버
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from model import kmeans_model, OUTPUT_DIR

# ── 앱 초기화 ──────────────────────────────────────────────────────────
app = FastAPI(
    title="K-Means Clustering Demo",
    description="NumPy만으로 구현한 K-Means 군집화 교육용 웹앱",
    version="1.0.0",
)

BASE_DIR = Path(__file__).parent
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── 스키마 ──────────────────────────────────────────────────────────────
class ClusterResponse(BaseModel):
    status: str
    iterations: int
    converged: bool
    wcss_history: list[float]
    final_wcss: float
    centroids: list[list[float]]
    cluster_plot: str
    loss_plot: str
    clustering_time_sec: float


class PredictRequest(BaseModel):
    x: float = Field(..., ge=-15, le=15, description="X 좌표")
    y: float = Field(..., ge=-15, le=15, description="Y 좌표")


class PredictResponse(BaseModel):
    x: float
    y: float
    cluster_id: int
    cluster_name: str
    distance: float
    prediction_plot: str


class StatusResponse(BaseModel):
    is_fitted: bool
    iterations_done: int
    converged: bool
    final_wcss: float | None
    output_files: list[str]


# ── 라우터 ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/cluster", response_model=ClusterResponse)
async def cluster(max_iter: int = Query(default=10, ge=1, le=10)):
    """K-Means 군집화 실행"""
    try:
        kmeans_model.generate_data()

        t0 = time.time()
        result = kmeans_model.fit(max_iter=max_iter)
        elapsed = round(time.time() - t0, 3)

        kmeans_model.save_cluster_plot()
        kmeans_model.save_loss_plot()

        return ClusterResponse(
            status="success",
            clustering_time_sec=elapsed,
            cluster_plot="/output/clustering_result.png",
            loss_plot="/output/loss_curve.png",
            **result,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    """(x, y) 좌표 → 클러스터 예측"""
    if not kmeans_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="모델이 학습되지 않았습니다. /cluster 먼저 호출하세요.",
        )
    try:
        cluster_id, distance = kmeans_model.predict(body.x, body.y)
        kmeans_model.save_prediction_plot(body.x, body.y, cluster_id)

        names = ["Cluster A", "Cluster B", "Cluster C"]
        return PredictResponse(
            x=body.x,
            y=body.y,
            cluster_id=cluster_id,
            cluster_name=names[cluster_id],
            distance=round(distance, 4),
            prediction_plot="/output/prediction_result.png",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def status():
    """모델 상태 확인"""
    files = (
        [f.name for f in OUTPUT_DIR.iterdir() if f.suffix == ".png"]
        if OUTPUT_DIR.exists()
        else []
    )
    return StatusResponse(
        is_fitted=kmeans_model.is_fitted,
        iterations_done=kmeans_model.iterations_done,
        converged=kmeans_model.converged,
        final_wcss=round(kmeans_model.final_wcss, 4) if kmeans_model.final_wcss else None,
        output_files=files,
    )


# ── 엔트리포인트 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
