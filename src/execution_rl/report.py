"""Funciones de visualización para la ejecución del algoritmo y evolución del entrenamiento."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .env import BestExecutionEnv


def plot_results(
    env: BestExecutionEnv,
    mode: str = "max",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Representa el precio (VWAP de mercado, VWAP del algoritmo y mid-price) y el volumen acumulado."""
    my_df, market_df, mpx = env.stats_df()
    my_vwap = my_df["vwap"]
    my_vol = my_df["vol"]
    vwap = market_df["vwap"]
    mkt_vol = market_df["vol"]
    norm_my_vol = (my_vol > 0) * 0.1
    if mode == "max":
        denom = mkt_vol.max()
    else:
        denom = mkt_vol.quantile(q=0.99)
    denom = denom if denom and denom > 0 else 1.0
    norm_mkt_vol = mkt_vol / denom
    norm_mkt_vol = norm_mkt_vol.where(norm_mkt_vol < 1, 1)

    mkt_sum = mkt_vol.sum()
    cum_mkt_vol = mkt_vol.cumsum() / (mkt_sum if mkt_sum > 0 else 1.0)
    cum_my_vol = my_vol.cumsum() / env.vol_care

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    ax1.set_title("Market VS Algo")
    ax1.plot(vwap, color="blue", label="vwap")
    ax1.plot(my_vwap, color="orange", label="algo")
    ax1.plot(mpx, color="red", label="mid_price")
    ax1.legend(["vwap", "algo", "mid_price"])
    ax1.grid(True)

    ax2.plot(cum_mkt_vol, color="blue", label="mkt cumvol")
    ax2.plot(cum_my_vol, color="orange", label="algo cumvol")
    ax2.bar(
        mkt_vol.index,
        norm_mkt_vol,
        width=0.00005,
        color="blue",
        alpha=0.6,
        label="mkt vol",
    )
    ax2.bar(
        my_vol.index,
        norm_my_vol,
        width=0.00007,
        color="orange",
        alpha=0.6,
        label="algo vol",
    )
    ax2.legend(["mkt cumvol", "algo cumvol", "mkt vol", "algo vol"])
    ax2.grid(True)
    fig.tight_layout()

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_training_rewards(
    history_rewards: list[float],
    window: int = 20,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Grafica la recompensa por episodio y su media móvil con ventana configurable."""
    series = pd.Series(history_rewards, dtype=float)
    rolling_mean = series.rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values, color="steelblue", alpha=0.35, label="Recompensa por episodio")
    ax.plot(
        rolling_mean.index,
        rolling_mean.values,
        color="darkblue",
        linewidth=2.0,
        label=f"Media móvil ({window} episodios)",
    )
    ax.set_title(f"Evolución de la recompensa de entrenamiento (media móvil {window} episodios)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa acumulada")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
