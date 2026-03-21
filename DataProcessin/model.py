"""
Data Preprocessing — Min-Max Normalization (NumPy Only)
sklearn 등 외부 ML 라이브러리 사용 금지
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=10)
    ax.set_ylabel(ylabel, color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#64748b")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#334155")
    ax.grid(True, color="#1e293b", alpha=0.6, linewidth=0.8)


def _legend(ax):
    leg = ax.legend(facecolor="#1e293b", edgecolor="#334155",
                    labelcolor="white", fontsize=9)
    leg.get_frame().set_alpha(0.85)


class PreprocessModel:
    def __init__(self):
        self.salary: np.ndarray | None = None
        self.age: np.ndarray | None = None
        self.salary_norm: np.ndarray | None = None
        self.age_norm: np.ndarray | None = None
        self.n_samples: int = 0
        self.is_processed: bool = False

    # ── 데이터 생성 ───────────────────────────────────────────────────
    def generate_data(self, n_samples: int = 50, seed: int = 42):
        """연봉(3천만~1억) + 나이(20~60) 데이터 생성"""
        np.random.seed(seed)
        self.n_samples = n_samples
        self.salary = np.random.uniform(30_000_000, 100_000_000, n_samples)
        self.age    = np.random.uniform(20, 60, n_samples)

    # ── Min-Max 정규화 ────────────────────────────────────────────────
    def normalize(self) -> dict:
        """Min-Max: (x - x_min) / (x_max - x_min) → 0~1"""
        self.salary_norm = (self.salary - self.salary.min()) / (self.salary.max() - self.salary.min())
        self.age_norm    = (self.age    - self.age.min())    / (self.age.max()    - self.age.min())
        self.is_processed = True

        return {
            "salary_stats": {
                "original":   self._stats(self.salary),
                "normalized": self._stats(self.salary_norm),
            },
            "age_stats": {
                "original":   self._stats(self.age),
                "normalized": self._stats(self.age_norm),
            },
        }

    def _stats(self, arr: np.ndarray) -> dict:
        return {
            "min":  round(float(arr.min()), 4),
            "max":  round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std()), 4),
        }

    # ── PNG 저장 ──────────────────────────────────────────────────────
    def save_scatter_comparison(self) -> str:
        """Before / After 산점도 비교"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0f172a")
        fig.suptitle("Min-Max Normalization: Before vs After",
                     color="white", fontsize=15, fontweight="bold", y=1.01)

        # Before
        ax1.set_facecolor("#0a1628")
        ax1.scatter(self.age, self.salary / 1_000_000,
                    c="#f59e0b", alpha=0.75, s=65, edgecolors="white", linewidths=0.3,
                    label=f"n={self.n_samples}")
        _style_ax(ax1, "Before Scaling  (Raw Data)", "Age (Years)", "Salary (Million ₩)")
        _legend(ax1)

        # After
        ax2.set_facecolor("#0a1628")
        ax2.scatter(self.age_norm, self.salary_norm,
                    c="#06b6d4", alpha=0.75, s=65, edgecolors="white", linewidths=0.3,
                    label=f"n={self.n_samples}")
        _style_ax(ax2, "After Min-Max Scaling", "Age (0~1)", "Salary (0~1)")
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_aspect("equal")
        _legend(ax2)

        plt.tight_layout()
        path = OUTPUT_DIR / "scatter_comparison.png"
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(path)

    def save_salary_hist(self) -> str:
        """연봉 정규화 전/후 히스토그램"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#0f172a")
        fig.suptitle("Salary Distribution: Before vs After Normalization",
                     color="white", fontsize=13, fontweight="bold")

        ax1.set_facecolor("#0a1628")
        ax1.hist(self.salary / 1_000_000, bins=15, color="#f59e0b",
                 alpha=0.8, edgecolor="#0f172a", linewidth=0.5)
        _style_ax(ax1, "Original Salary", "Salary (Million ₩)", "Count")

        ax2.set_facecolor("#0a1628")
        ax2.hist(self.salary_norm, bins=15, color="#06b6d4",
                 alpha=0.8, edgecolor="#0f172a", linewidth=0.5)
        # 0~1 범위 표시선
        ax2.axvline(0, color="#f43f5e", linestyle="--", linewidth=1.2, alpha=0.7, label="min=0")
        ax2.axvline(1, color="#22c55e", linestyle="--", linewidth=1.2, alpha=0.7, label="max=1")
        _style_ax(ax2, "Normalized Salary (0~1)", "Salary (0~1)", "Count")
        _legend(ax2)

        plt.tight_layout()
        path = OUTPUT_DIR / "salary_hist.png"
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(path)

    def save_age_hist(self) -> str:
        """나이 정규화 전/후 히스토그램"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#0f172a")
        fig.suptitle("Age Distribution: Before vs After Normalization",
                     color="white", fontsize=13, fontweight="bold")

        ax1.set_facecolor("#0a1628")
        ax1.hist(self.age, bins=15, color="#8b5cf6",
                 alpha=0.8, edgecolor="#0f172a", linewidth=0.5)
        _style_ax(ax1, "Original Age", "Age (Years)", "Count")

        ax2.set_facecolor("#0a1628")
        ax2.hist(self.age_norm, bins=15, color="#f43f5e",
                 alpha=0.8, edgecolor="#0f172a", linewidth=0.5)
        ax2.axvline(0, color="#f59e0b", linestyle="--", linewidth=1.2, alpha=0.7, label="min=0")
        ax2.axvline(1, color="#22c55e", linestyle="--", linewidth=1.2, alpha=0.7, label="max=1")
        _style_ax(ax2, "Normalized Age (0~1)", "Age (0~1)", "Count")
        _legend(ax2)

        plt.tight_layout()
        path = OUTPUT_DIR / "age_hist.png"
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(path)


# 싱글톤 인스턴스
preprocess_model = PreprocessModel()
