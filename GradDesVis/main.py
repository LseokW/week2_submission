"""
FastAPI — Gradient Descent Visualization Web App
f(x) = x² 경사 하강법 시각화 교육용 서버
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from model import gd_model, OUTPUT_DIR

app = FastAPI(
    title="Gradient Descent Visualization",
    description="f(x)=x² 경사 하강법 교육용 웹앱",
    version="1.0.0",
)

BASE_DIR = Path(__file__).parent
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── 스키마 ──────────────────────────────────────────────────────────
class StepRecord(BaseModel):
    step: int
    x: float
    loss: float
    gradient: float
    delta_x: float

class RunResponse(BaseModel):
    status: str
    x0: float
    lr: float
    steps: int
    final_x: float
    final_loss: float
    converged: bool
    diverged: bool
    actual_steps: int
    history: list[StepRecord]
    landscape_plot: str
    loss_curve: str
    run_time_sec: float

class StatusResponse(BaseModel):
    is_run: bool
    converged: bool
    diverged: bool
    final_x: float | None
    final_loss: float | None
    output_files: list[str]


# ── 라우터 ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/run", response_model=RunResponse)
async def run(
    x0: float    = Query(default=-4.0, ge=-5.0, le=5.0),
    lr: float    = Query(default=0.1,  ge=0.01, le=0.9),
    steps: int   = Query(default=20,   ge=5,    le=50),
):
    """경사 하강법 실행"""
    try:
        t0 = time.time()
        result = gd_model.run(x0=x0, lr=lr, steps=steps)
        gd_model.save_landscape_plot()
        gd_model.save_loss_curve()
        elapsed = round(time.time() - t0, 3)

        return RunResponse(
            status="diverged" if result["diverged"] else "success",
            x0=x0,
            lr=lr,
            steps=steps,
            final_x=result["final_x"],
            final_loss=result["final_loss"],
            converged=result["converged"],
            diverged=result["diverged"],
            actual_steps=len(result["history"]),
            history=[StepRecord(**h) for h in result["history"]],
            landscape_plot="/output/landscape_plot.png",
            loss_curve="/output/loss_curve.png",
            run_time_sec=elapsed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def status():
    files = [f.name for f in OUTPUT_DIR.iterdir() if f.suffix == ".png"] \
            if OUTPUT_DIR.exists() else []
    return StatusResponse(
        is_run=gd_model.is_run,
        converged=gd_model.converged,
        diverged=gd_model.diverged,
        final_x=gd_model.final_x,
        final_loss=gd_model.final_loss,
        output_files=files,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
