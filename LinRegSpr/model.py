"""
Hooke's Law TensorFlow Model
훅의 법칙을 학습하는 TensorFlow 선형회귀 모델
F = k*x  =>  Length = k * mass + L0
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import tensorflow as tf

# ─── 출력 디렉토리 ───────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 다크 테마 설정 ─────────────────────────────────────────────────
DARK_BG     = "#0f172a"
DARK_PANEL  = "#1e293b"
DARK_BORDER = "#334155"
CYAN        = "#06b6d4"
VIOLET      = "#8b5cf6"
ROSE        = "#f43f5e"
AMBER       = "#f59e0b"
GREEN       = "#22c55e"
WHITE       = "#f1f5f9"
GRAY        = "#94a3b8"

def _apply_dark_style(fig, ax_list):
    """공통 다크 테마 적용"""
    fig.patch.set_facecolor(DARK_BG)
    for ax in ax_list:
        ax.set_facecolor(DARK_PANEL)
        ax.tick_params(colors=GRAY, labelsize=11)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_BORDER)
        ax.grid(color=DARK_BORDER, linestyle='--', linewidth=0.6, alpha=0.7)


# ─── 데이터 생성 ─────────────────────────────────────────────────────
TRUE_K  = 2.0     # cm/kg  (용수철 상수 효과)
TRUE_L0 = 10.0    # cm     (초기 길이)
NOISE_STD = 1.2   # cm     (측정 노이즈 표준편차)

def make_dataset(n_points: int = 20, seed: int = 42):
    """훅의 법칙 시뮬레이션 데이터셋 생성"""
    rng = np.random.RandomState(seed)
    masses = np.linspace(0, 10, n_points, dtype=np.float32)
    true_lengths = TRUE_K * masses + TRUE_L0
    noise = rng.normal(0, NOISE_STD, size=n_points).astype(np.float32)
    measured_lengths = true_lengths + noise
    return masses, true_lengths, measured_lengths


# ─── TensorFlow 모델 ─────────────────────────────────────────────────
class SpringModel:
    """훅의 법칙 선형회귀 TensorFlow 모델"""

    def __init__(self, learning_rate: float = 0.1):
        self.lr = learning_rate
        self.model: tf.keras.Model | None = None
        self.history = None
        self.is_trained = False
        self.masses, self.true_lengths, self.measured_lengths = make_dataset()

    def build(self):
        """Dense(1) 단층 선형 모델 구성"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=1,
                input_shape=[1],
                kernel_initializer='random_normal',
                bias_initializer='zeros',
                name='linear_layer'
            )
        ], name='hookes_law_model')
        self.compile()

    def compile(self):
        """Adam optimizer, MSE loss 컴파일"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mean_squared_error',
            metrics=['mae']
        )

    def train(self, epochs: int = 500) -> dict:
        """모델 학습 수행"""
        if self.model is None:
            self.build()

        self.history = self.model.fit(
            self.masses,
            self.measured_lengths,
            epochs=epochs,
            verbose=0,
            batch_size=len(self.masses)
        )
        self.is_trained = True

        # 학습된 파라미터 추출
        weights = self.model.layers[0].get_weights()
        learned_k  = float(weights[0][0][0])
        learned_b  = float(weights[1][0])
        final_loss = float(self.history.history['loss'][-1])
        final_mae  = float(self.history.history['mae'][-1])

        return {
            "epochs": epochs,
            "final_loss": round(final_loss, 6),
            "final_mae": round(final_mae, 4),
            "learned_k": round(learned_k, 4),
            "learned_b": round(learned_b, 4),
            "true_k": TRUE_K,
            "true_b": TRUE_L0,
        }

    def predict(self, mass: float) -> float:
        """단일 질량값으로 길이 예측"""
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다. /train 먼저 호출하세요.")
        result = self.model.predict(np.array([[mass]], dtype=np.float32), verbose=0)
        return float(result[0][0])

    # ─── 시각화 메서드 ────────────────────────────────────────────────

    def save_loss_plot(self, path: str | None = None) -> str:
        """Epoch별 Loss 그래프 저장"""
        if self.history is None:
            raise RuntimeError("학습 history가 없습니다.")

        losses = self.history.history['loss']
        maes   = self.history.history['mae']
        epochs = range(1, len(losses) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Training History — Hooke\'s Law Model',
                     color=WHITE, fontsize=16, fontweight='bold', y=1.02)
        _apply_dark_style(fig, axes)

        # Loss subplot
        ax1 = axes[0]
        ax1.plot(epochs, losses, color=CYAN, linewidth=2.5, label='MSE Loss')
        ax1.fill_between(epochs, losses, alpha=0.15, color=CYAN)
        ax1.set_title('Loss (MSE) per Epoch', fontsize=13, pad=10)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Mean Squared Error', fontsize=12)
        ax1.set_yscale('log')
        ax1.legend(facecolor=DARK_PANEL, edgecolor=DARK_BORDER,
                   labelcolor=WHITE, fontsize=11)
        # 최솟값 표시
        min_idx = np.argmin(losses)
        ax1.annotate(
            f'  Final: {losses[-1]:.4f}',
            xy=(epochs[-1], losses[-1]),
            color=AMBER, fontsize=10,
            xytext=(epochs[-1]*0.65, losses[-1]*3),
            arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.5)
        )

        # MAE subplot
        ax2 = axes[1]
        ax2.plot(epochs, maes, color=VIOLET, linewidth=2.5, label='MAE')
        ax2.fill_between(epochs, maes, alpha=0.15, color=VIOLET)
        ax2.set_title('MAE per Epoch', fontsize=13, pad=10)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Mean Absolute Error (cm)', fontsize=12)
        ax2.set_yscale('log')
        ax2.annotate(
            f'  Final: {maes[-1]:.4f}',
            xy=(epochs[-1], maes[-1]),
            color=AMBER, fontsize=10,
            xytext=(epochs[-1]*0.65, maes[-1]*3),
            arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.5)
        )
        ax2.legend(facecolor=DARK_PANEL, edgecolor=DARK_BORDER,
                   labelcolor=WHITE, fontsize=11)

        fig.tight_layout(pad=2.0)
        save_path = path or os.path.join(OUTPUT_DIR, "loss_curve.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=DARK_BG, edgecolor='none')
        plt.close(fig)
        return save_path

    def save_regression_plot(self, path: str | None = None) -> str:
        """훈련 데이터 + 회귀선 시각화 저장"""
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        x_range = np.linspace(-0.5, 11.0, 200, dtype=np.float32)
        predicted = self.model.predict(x_range.reshape(-1, 1), verbose=0).flatten()

        fig, ax = plt.subplots(figsize=(10, 6))
        _apply_dark_style(fig, [ax])
        fig.suptitle('Hooke\'s Law — Spring Length Regression',
                     color=WHITE, fontsize=16, fontweight='bold')

        # 이상적 직선 (설계: 파란 점선)
        ax.plot(x_range, TRUE_K * x_range + TRUE_L0,
                color='#3b82f6', linewidth=1.8, linestyle='--',
                label=f'Ideal: L = {TRUE_K}m + {TRUE_L0}', alpha=0.8)

        # 측정 데이터
        ax.scatter(self.masses, self.measured_lengths,
                   color=CYAN, s=80, zorder=5, alpha=0.9,
                   edgecolors='white', linewidths=0.5,
                   label='Measured Data (w/ noise)')

        # TF 예측선
        weights = self.model.layers[0].get_weights()
        k_fit = float(weights[0][0][0])
        b_fit = float(weights[1][0])
        ax.plot(x_range, predicted,
                color=ROSE, linewidth=2.5,
                label=f'TF Model: L = {k_fit:.3f}m + {b_fit:.3f}',
                path_effects=[pe.Stroke(linewidth=4, foreground=DARK_BG), pe.Normal()])

        ax.set_xlabel('Mass (kg)', fontsize=13)
        ax.set_ylabel('Spring Length (cm)', fontsize=13)
        ax.set_xlim(-0.5, 11)
        ax.legend(facecolor=DARK_PANEL, edgecolor=DARK_BORDER,
                  labelcolor=WHITE, fontsize=11, loc='upper left')

        # 물리 공식 텍스트
        ax.text(0.97, 0.08,
                'F = k·x\nL = k·m + L₀',
                transform=ax.transAxes,
                ha='right', va='bottom',
                color=AMBER, fontsize=13, alpha=0.85,
                fontfamily='monospace')

        fig.tight_layout()
        save_path = path or os.path.join(OUTPUT_DIR, "spring_regression.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=DARK_BG, edgecolor='none')
        plt.close(fig)
        return save_path

    def save_prediction_plot(self, mass: float, pred_length: float,
                              path: str | None = None) -> str:
        """예측 결과 강조 시각화 저장"""
        x_range = np.linspace(-0.5, max(12.0, mass + 2.0), 200, dtype=np.float32)
        predicted_line = self.model.predict(x_range.reshape(-1, 1), verbose=0).flatten()

        fig, ax = plt.subplots(figsize=(10, 6))
        _apply_dark_style(fig, [ax])
        fig.suptitle(f'Prediction: Mass = {mass:.1f} kg → Length = {pred_length:.2f} cm',
                     color=WHITE, fontsize=15, fontweight='bold')

        # 배경 데이터
        ax.scatter(self.masses, self.measured_lengths,
                   color=CYAN, s=60, alpha=0.5,
                   edgecolors='white', linewidths=0.3,
                   label='Training Data', zorder=3)

        # 회귀선
        ax.plot(x_range, predicted_line,
                color=ROSE, linewidth=2.0, alpha=0.8,
                label='Regression Line', zorder=4)

        # 예측점 강조
        ax.scatter([mass], [pred_length],
                   marker='*', s=400, color=AMBER, zorder=10,
                   edgecolors='white', linewidths=0.8,
                   label=f'Prediction ({mass:.1f} kg, {pred_length:.2f} cm)')

        # 점선 가이드라인
        ax.axvline(mass, color=AMBER, linestyle=':', linewidth=1.2, alpha=0.6)
        ax.axhline(pred_length, color=AMBER, linestyle=':', linewidth=1.2, alpha=0.6)

        # 레이블 박스
        ax.annotate(
            f'{pred_length:.2f} cm',
            xy=(mass, pred_length),
            xytext=(mass + 0.5, pred_length + 1.5),
            color=AMBER, fontsize=13, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=AMBER, lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=DARK_PANEL,
                      edgecolor=AMBER, alpha=0.9)
        )

        ax.set_xlabel('Mass (kg)', fontsize=13)
        ax.set_ylabel('Spring Length (cm)', fontsize=13)
        ax.legend(facecolor=DARK_PANEL, edgecolor=DARK_BORDER,
                  labelcolor=WHITE, fontsize=11)

        fig.tight_layout()
        save_path = path or os.path.join(OUTPUT_DIR, "prediction_result.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=DARK_BG, edgecolor='none')
        plt.close(fig)
        return save_path


# ─── 전역 모델 인스턴스 (FastAPI 공유) ──────────────────────────────
spring_model = SpringModel(learning_rate=0.1)
