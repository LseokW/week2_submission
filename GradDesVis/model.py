"""
Gradient Descent Visualization — NumPy Only
f(x) = x², gradient = 2x
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DIVERGE_THRESHOLD = 50.0
CONVERGE_THRESHOLD = 1e-6


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=10)
    ax.set_ylabel(ylabel, color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#64748b")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#334155")
    ax.grid(True, color="#1e293b", alpha=0.6, linewidth=0.8)


class GradientDescentModel:
    def __init__(self):
        self.history: list[dict] = []
        self.final_x: float | None = None
        self.final_loss: float | None = None
        self.converged: bool = False
        self.diverged: bool = False
        self.is_run: bool = False
        self.params: dict = {}

    # ── 손실 함수 & Gradient ────────────────────────────────────────
    @staticmethod
    def loss(x: float) -> float:
        return float(x ** 2)

    @staticmethod
    def gradient(x: float) -> float:
        return float(2 * x)

    # ── 경사 하강법 실행 ─────────────────────────────────────────────
    def run(self, x0: float = -4.0, lr: float = 0.1, steps: int = 20) -> dict:
        self.params = {"x0": x0, "lr": lr, "steps": steps}
        self.history = []
        self.converged = False
        self.diverged = False

        x = x0
        for i in range(1, steps + 1):
            loss_val = self.loss(x)
            grad_val = self.gradient(x)

            self.history.append({
                "step": i,
                "x": round(x, 6),
                "loss": round(loss_val, 6),
                "gradient": round(grad_val, 6),
                "delta_x": round(-lr * grad_val, 6),
            })

            x_new = x - lr * grad_val

            # 발산 체크
            if abs(x_new) > DIVERGE_THRESHOLD or not np.isfinite(x_new):
                self.diverged = True
                x = x_new
                break

            x = x_new

            # 수렴 체크
            if self.loss(x) < CONVERGE_THRESHOLD:
                self.converged = True
                break

        self.final_x = round(float(x), 8)
        self.final_loss = round(self.loss(float(x)), 8)
        self.is_run = True

        return {
            "final_x": self.final_x,
            "final_loss": self.final_loss,
            "converged": self.converged,
            "diverged": self.diverged,
            "history": self.history,
        }

    # ── PNG: Loss Landscape + 이동 경로 ────────────────────────────
    def save_landscape_plot(self) -> str:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0a1628")

        # y=x² 곡선 범위
        xs = [h["x"] for h in self.history]
        x_span = max(6.0, abs(self.params["x0"]) + 2)
        x_rng = np.linspace(-x_span, x_span, 300)
        y_rng = x_rng ** 2

        # Loss 함수 곡선
        ax.plot(x_rng, y_rng, color="#8b5cf6", linewidth=2.5,
                label="f(x) = x²", zorder=2)

        # 최솟값 수직선
        ax.axvline(0, color="#22c55e", linestyle="--",
                   linewidth=1.2, alpha=0.6, label="Global min (x=0)")

        if self.history:
            xs_arr = np.array([h["x"] for h in self.history])
            ys_arr = np.array([h["loss"] for h in self.history])

            # y 클램핑 (발산 시 그래프 폭발 방지)
            y_max_vis = max(self.params["x0"] ** 2 * 1.3, 30.0)
            ys_clamped = np.clip(ys_arr, 0, y_max_vis)

            # 이동 경로 점선
            ax.plot(xs_arr, ys_clamped, "--",
                    color="#f43f5e", alpha=0.55, linewidth=1.5, zorder=3)

            # 이동 경로 점
            ax.scatter(xs_arr, ys_clamped,
                       c="#f43f5e", s=70, zorder=5,
                       edgecolors="white", linewidths=0.5)

            # 시작점 ★
            ax.scatter([xs_arr[0]], [ys_clamped[0]],
                       c="#f59e0b", s=280, marker="*", zorder=6,
                       edgecolors="white", linewidths=1.2, label="Start")

            # 끝점 어노테이션
            end_x, end_y = float(xs_arr[-1]), float(ys_clamped[-1])
            if self.converged:
                ax.annotate("✓ Converged",
                            xy=(end_x, end_y),
                            xytext=(end_x + x_span * 0.2, end_y + y_max_vis * 0.1),
                            color="#22c55e", fontsize=9, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.2))
            elif self.diverged:
                ax.text(0, y_max_vis * 0.7, "⚠ DIVERGED",
                        color="#f43f5e", fontsize=18, fontweight="black",
                        ha="center", alpha=0.35, style="italic")

            ax.set_ylim(-1, y_max_vis)

        _style_ax(ax, f"Gradient Descent on f(x) = x²  (lr={self.params['lr']}, x₀={self.params['x0']})",
                  "Parameter x", "Loss f(x)")

        leg = ax.legend(facecolor="#1e293b", edgecolor="#334155",
                        labelcolor="white", fontsize=9)
        leg.get_frame().set_alpha(0.85)

        path = OUTPUT_DIR / "landscape_plot.png"
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(path)

    # ── PNG: Step별 Loss 수렴 곡선 ──────────────────────────────────
    def save_loss_curve(self) -> str:
        steps = [h["step"] for h in self.history]
        losses = [h["loss"] for h in self.history]

        fig, ax = plt.subplots(figsize=(8, 3.8))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0a1628")

        # y 클램핑
        y_max_vis = max(losses[0] * 1.2 if losses else 30, 30) if losses else 30
        losses_clamped = [min(l, y_max_vis) for l in losses]

        # Fill
        ax.fill_between(steps, losses_clamped, alpha=0.12, color="#06b6d4")

        # Line
        ax.plot(steps, losses_clamped, color="#06b6d4",
                linewidth=2.5, zorder=3)

        # Scatter
        ax.scatter(steps, losses_clamped,
                   c="#8b5cf6", s=55, zorder=4,
                   edgecolors="white", linewidths=0.6)

        # 수렴 어노테이션
        if self.converged and losses:
            ax.annotate(f"Converged @ step {steps[-1]}",
                        xy=(steps[-1], losses_clamped[-1]),
                        xytext=(steps[-1] - len(steps) * 0.3,
                                losses_clamped[-1] + y_max_vis * 0.15),
                        color="#22c55e", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.2))

        if self.diverged and len(losses) > 1:
            ax.text(steps[len(steps)//2], y_max_vis * 0.75,
                    "⚠ Loss Diverging",
                    color="#f43f5e", fontsize=11, fontweight="bold",
                    ha="center", alpha=0.7)

        ax.set_xticks(steps[::max(1, len(steps)//10)])
        ax.set_ylim(-y_max_vis * 0.05, y_max_vis)
        _style_ax(ax, "Loss per Step", "Step", "Loss f(x) = x²")

        path = OUTPUT_DIR / "loss_curve.png"
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(path)


# 싱글톤
gd_model = GradientDescentModel()
