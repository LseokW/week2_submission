"""
K-Means Clustering — NumPy Only Implementation
sklearn 등 외부 ML 라이브러리 사용 금지
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Cluster color palette (matches UI)
COLORS = ["#06b6d4", "#8b5cf6", "#f43f5e"]
CLUSTER_NAMES = ["Cluster A", "Cluster B", "Cluster C"]


class KMeansModel:
    def __init__(self, k: int = 3):
        self.k = k
        self.data: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.wcss_history: list[float] = []
        self.is_fitted: bool = False
        self.iterations_done: int = 0
        self.converged: bool = False
        self.final_wcss: float | None = None

    # ── 데이터 생성 ──────────────────────────────────────────────────
    def generate_data(self, seed: int = 42) -> np.ndarray:
        """3그룹 × 30개, 가우시안 분포 2D 데이터 생성"""
        np.random.seed(seed)
        centers = [(0.0, 0.0), (6.0, 6.0), (-6.0, 6.0)]
        std = 1.5
        groups = []
        for cx, cy in centers:
            pts = np.random.randn(30, 2) * std + np.array([cx, cy])
            groups.append(pts)
        self.data = np.vstack(groups)
        return self.data

    # ── K-Means 핵심 알고리즘 ────────────────────────────────────────
    def fit(self, max_iter: int = 10) -> dict:
        """K-Means 학습: 초기화 → 배정 → 이동 → 수렴 체크"""
        if self.data is None:
            self.generate_data()

        # 초기화: 데이터에서 K개 랜덤 선택
        np.random.seed(0)
        idx = np.random.choice(len(self.data), self.k, replace=False)
        self.centroids = self.data[idx].copy()

        self.wcss_history = []
        self.converged = False

        for i in range(max_iter):
            # 1. 배정: 각 점 → 가장 가까운 중심점
            self.labels = self._assign()

            # 2. WCSS 계산
            self.wcss_history.append(self._wcss())

            # 3. 중심점 이동
            new_centroids = self._update_centroids()

            # 4. 수렴 체크: 중심점 변화량
            shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids

            if shift < 1e-6:
                self.converged = True
                self.iterations_done = i + 1
                break
        else:
            self.iterations_done = max_iter

        self.final_wcss = self.wcss_history[-1]
        self.is_fitted = True

        return {
            "iterations": self.iterations_done,
            "converged": self.converged,
            "wcss_history": [round(w, 4) for w in self.wcss_history],
            "final_wcss": round(self.final_wcss, 4),
            "centroids": self.centroids.tolist(),
        }

    def _assign(self) -> np.ndarray:
        """유클리드 거리로 가장 가까운 클러스터에 배정"""
        # (N, 1, 2) - (1, K, 2) → (N, K, 2) → (N, K)
        diffs = self.data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        return np.argmin(dists, axis=1)

    def _update_centroids(self) -> np.ndarray:
        """각 클러스터의 평균으로 중심점 갱신"""
        new_c = np.zeros_like(self.centroids)
        for k in range(self.k):
            members = self.data[self.labels == k]
            if len(members) > 0:
                new_c[k] = members.mean(axis=0)
            else:
                new_c[k] = self.centroids[k]  # 빈 클러스터 유지
        return new_c

    def _wcss(self) -> float:
        """WCSS: 각 점과 배정된 중심점 사이의 거리 제곱합"""
        total = 0.0
        for k in range(self.k):
            members = self.data[self.labels == k]
            if len(members) > 0:
                diff = members - self.centroids[k]
                total += (diff ** 2).sum()
        return float(total)

    # ── 예측 ─────────────────────────────────────────────────────────
    def predict(self, x: float, y: float) -> tuple[int, float]:
        """입력 좌표 → 가장 가까운 클러스터 인덱스 + 거리"""
        pt = np.array([[x, y]])
        diffs = pt - self.centroids
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cluster_id = int(np.argmin(dists))
        distance = float(dists[cluster_id])
        return cluster_id, distance

    # ── PNG 저장 ──────────────────────────────────────────────────────
    def save_cluster_plot(self) -> str:
        """군집 결과 플롯 저장"""
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0a1628")

        # 데이터 포인트
        for k in range(self.k):
            mask = self.labels == k
            ax.scatter(
                self.data[mask, 0], self.data[mask, 1],
                c=COLORS[k], alpha=0.75, s=70,
                label=CLUSTER_NAMES[k], edgecolors="white", linewidths=0.3,
                zorder=2
            )

        # 중심점 (별★)
        ax.scatter(
            self.centroids[:, 0], self.centroids[:, 1],
            c=COLORS[:self.k], s=350, marker="*",
            edgecolors="white", linewidths=1.5,
            zorder=5, label="Centroids"
        )

        # 중심점 보이론서클
        for k in range(self.k):
            circle = plt.Circle(
                (self.centroids[k, 0], self.centroids[k, 1]),
                0.4, color=COLORS[k], fill=False, alpha=0.5, linewidth=1.2
            )
            ax.add_patch(circle)

        _style_ax(ax, "K-Means Clustering Result  (K=3)", "Feature X", "Feature Y")
        _legend(ax)

        path = OUTPUT_DIR / "clustering_result.png"
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(path)

    def save_loss_plot(self) -> str:
        """WCSS Loss 곡선 저장"""
        iters = list(range(1, len(self.wcss_history) + 1))

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0a1628")

        # Area fill
        ax.fill_between(iters, self.wcss_history, alpha=0.12, color="#06b6d4")

        # Line
        ax.plot(iters, self.wcss_history, color="#06b6d4",
                linewidth=2.5, zorder=3)

        # Scatter points
        ax.scatter(iters, self.wcss_history, c="#8b5cf6",
                   s=80, zorder=4, edgecolors="white", linewidths=0.8)

        # Convergence annotation
        if self.converged:
            last_i = iters[-1]
            last_w = self.wcss_history[-1]
            ax.annotate(
                f"Converged @ iter {last_i}",
                xy=(last_i, last_w),
                xytext=(last_i - 0.8, last_w + (max(self.wcss_history) - last_w) * 0.3),
                color="#22c55e", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.2)
            )

        ax.set_xticks(iters)
        _style_ax(ax, "WCSS Loss per Iteration", "Iteration", "WCSS (Loss)")

        path = OUTPUT_DIR / "loss_curve.png"
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(path)

    def save_prediction_plot(self, x: float, y: float, cluster_id: int) -> str:
        """예측 포인트가 강조된 군집 플롯 저장"""
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0a1628")

        # 기존 클러스터 (흐리게)
        for k in range(self.k):
            mask = self.labels == k
            ax.scatter(
                self.data[mask, 0], self.data[mask, 1],
                c=COLORS[k], alpha=0.40, s=55,
                edgecolors="white", linewidths=0.2, zorder=2
            )

        # 중심점
        ax.scatter(
            self.centroids[:, 0], self.centroids[:, 1],
            c=COLORS[:self.k], s=280, marker="*",
            edgecolors="white", linewidths=1.5, zorder=4
        )

        # 예측 포인트 (금색 별)
        ax.scatter(
            [x], [y], c="#f59e0b", s=520, marker="*",
            edgecolors="white", linewidths=1.8, zorder=6,
            label=f"Prediction → {CLUSTER_NAMES[cluster_id]}"
        )

        # 예측 포인트 ~ 중심점 연결선
        cx, cy = self.centroids[cluster_id]
        ax.plot([x, cx], [y, cy], "--",
                color=COLORS[cluster_id], alpha=0.6, linewidth=1.5, zorder=3)

        ax.annotate(
            f"→ {CLUSTER_NAMES[cluster_id]}",
            xy=(x, y), xytext=(x + 0.5, y + 0.5),
            color="#f59e0b", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.2)
        )

        _style_ax(ax, "Prediction Result", "Feature X", "Feature Y")
        _legend(ax)

        path = OUTPUT_DIR / "prediction_result.png"
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(path)


# ── 헬퍼 ──────────────────────────────────────────────────────────────
def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=11)
    ax.set_ylabel(ylabel, color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#64748b")
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#1e293b", alpha=0.6, linewidth=0.8)


def _legend(ax):
    leg = ax.legend(
        facecolor="#1e293b", edgecolor="#334155",
        labelcolor="white", fontsize=9
    )
    leg.get_frame().set_alpha(0.8)


# 싱글톤 모델 인스턴스
kmeans_model = KMeansModel(k=3)
