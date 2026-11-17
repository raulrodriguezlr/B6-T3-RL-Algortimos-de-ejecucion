"""Evaluación de políticas sobre un panel fijo de episodios y métricas frente al plan TWAP."""

import numpy as np
import pandas as pd

from .agents import DDQNAgent
from .env import BestExecutionEnv


def run_episode(
    env: BestExecutionEnv,
    agent: DDQNAgent,
    seed: int | None = None,
    off_plan_threshold: float = 1.0,
) -> dict[str, float]:
    """Ejecuta un episodio sin exploración y devuelve métricas de seguimiento del plan y de precio."""
    prev_env, prev_trainable = agent.env, agent._trainable
    agent.env = env
    agent.set_trainable(False)
    devs: list[float] = []
    cum_reward = 0.0
    try:
        s = env.reset(seed=seed)
        while True:
            a = agent.act(s)
            s, r, done, _ = env.step(a)
            cum_reward += r
            # desvío frente al plan del entorno (lineal hasta target_horizon)
            devs.append(env.exec_vol - env._plan_volume(env.state_pos))
            if done:
                break
    finally:
        agent.env = prev_env
        agent.set_trainable(prev_trainable)

    abs_dev = np.abs(np.asarray(devs))
    algo_vwap = env._compute_algo_vwap()
    mkt_vwap = float(env.episode_vwap)
    # compra: slippage positivo si se compra por debajo del VWAP de mercado
    slippage = (mkt_vwap - algo_vwap) / mkt_vwap * 1e4 if env.exec_vol > 0 else np.nan
    # barrido final: lo pendiente se compra al ask1 del último bin, para comparar órdenes incompletas
    pending = env.vol_care - env.exec_vol
    spent = (algo_vwap if env.exec_vol > 0 else 0.0) * env.exec_vol
    cleanup_vwap = (spent + pending * float(env._ask[-1])) / env.vol_care
    cleanup_slippage = (mkt_vwap - cleanup_vwap) / mkt_vwap * 1e4
    mid =(env.episode["ask1"] + env.episode["bid1"]) / 2.0
    half_spread = float(((env.episode["ask1"] - env.episode["bid1"]) / 2.0 / mid * 1e4).mean())
    return {
        "seed": float(seed) if seed is not None else np.nan,
        "day": str(env.episode.index[0].date()),
        "start": str(env.episode.index[0].time()),
        "episode_bins": float(env.episode_bins),
        "vol_care": float(env.vol_care),
        "exec_vol": float(env.exec_vol),
        "completed_pct": 100.0 * env.exec_vol / env.vol_care,
        "time_used_pct": 100.0 * env.state_pos / env.episode_bins,
        "mean_abs_dev": float(abs_dev.mean()),
        "max_abs_dev": float(abs_dev.max()),
        "off_plan_steps_pct": 100.0 * float((abs_dev > off_plan_threshold).mean()),
        "cum_reward": float(cum_reward),
        "slippage_bps": float(slippage),
        "cleanup_slippage_bps": float(cleanup_slippage),
        "half_spread_bps": half_spread,
    }


def evaluate_panel(
    env: BestExecutionEnv,
    agent: DDQNAgent,
    seeds: list[int],
) -> pd.DataFrame:
    """Evalúa la política sobre episodios reproducibles (un seed por episodio) y devuelve una fila por episodio."""
    return pd.DataFrame([run_episode(env, agent, seed=s) for s in seeds])


def summarize_panel(df: pd.DataFrame) -> dict[str, float]:
    """Resume un panel con medias y la dispersión del slippage."""
    return {
        "mean_abs_dev": float(df["mean_abs_dev"].mean()),
        "max_abs_dev": float(df["max_abs_dev"].max()),
        "off_plan_steps_pct": float(df["off_plan_steps_pct"].mean()),
        "completed_pct": float(df["completed_pct"].mean()),
        "time_used_pct": float(df["time_used_pct"].mean()),
        "slippage_bps_mean": float(df["slippage_bps"].mean()),
        "slippage_bps_std": float(df["slippage_bps"].std()),
        "cum_reward": float(df["cum_reward"].mean()),
    }


def episode_trace(env: BestExecutionEnv, agent: DDQNAgent, seed: int) -> pd.DataFrame:
    """Recorre un episodio greedy y devuelve, por step, volumen ejecutado, plan y acción."""
    run_episode(env, agent, seed=seed)
    n = env.state_pos
    steps = np.arange(1, n + 1)
    return pd.DataFrame(
        {
            "step": steps,
            "time": env.episode.index[:n],
            "action": env.actions_hist[:n],
            "exec_vol": np.cumsum(env.vol_hist[:n]),
            "plan": [env._plan_volume(k) for k in steps],
            "vol_care": env.vol_care,
            "episode_bins": env.episode_bins,
        }
    )
