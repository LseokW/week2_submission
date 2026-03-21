"""
FastAPI — Data Preprocessing Educational Web App
Min-Max 정규화를 numpy만으로 구현한 교육용 웹앱
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from model import preprocess_model, OUTPUT_DIR

app = FastAPI(
    title="Data Preprocessing Demo",
    description="Min-Max 정규화 교육용 웹앱",
    version="1.0.0",
)

BASE_DIR = Path(__file__).parent
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── 스키마 ──────────────────────────────────────────────────────────
class StatsDetail(BaseModel):
    min: float
    max: float
    mean: float
    std: float

class FeatureStats(BaseModel):
    original: StatsDetail
    normalized: StatsDetail

class NormalizeResponse(BaseModel):
    status: str
    n_samples: int
    salary_stats: FeatureStats
    age_stats: FeatureStats
    scatter_plot: str
    salary_hist: str
    age_hist: str
    processing_time_sec: float

class StatusResponse(BaseModel):
    is_processed: bool
    n_samples: int
    output_files: list[str]


# ── 라우터 ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/normalize", response_model=NormalizeResponse)
async def normalize(n_samples: int = Query(default=50, ge=10, le=100)):
    """Min-Max 정규화 실행 + PNG 3종 생성"""
    try:
        t0 = time.time()
        preprocess_model.generate_data(n_samples=n_samples)
        result = preprocess_model.normalize()
        preprocess_model.save_scatter_comparison()
        preprocess_model.save_salary_hist()
        preprocess_model.save_age_hist()
        elapsed = round(time.time() - t0, 3)

        return NormalizeResponse(
            status="success",
            n_samples=n_samples,
            salary_stats=FeatureStats(**result["salary_stats"]),
            age_stats=FeatureStats(**result["age_stats"]),
            scatter_plot="/output/scatter_comparison.png",
            salary_hist="/output/salary_hist.png",
            age_hist="/output/age_hist.png",
            processing_time_sec=elapsed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def status():
    files = [f.name for f in OUTPUT_DIR.iterdir() if f.suffix == ".png"] \
            if OUTPUT_DIR.exists() else []
    return StatusResponse(
        is_processed=preprocess_model.is_processed,
        n_samples=preprocess_model.n_samples,
        output_files=files,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
