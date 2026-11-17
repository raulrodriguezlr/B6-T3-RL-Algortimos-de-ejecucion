"""Configuración y ejecución reproducible de experimentos (una semilla por proceso, en paralelo)."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .agents import TWAP, DDQNAgent
from .data import load_market_data, set_seeds
from .env import BestExecutionEnv
from .evaluation import evaluate_panel, summarize_panel
from .exploration import AdaptiveEpsilon, LinearEpsilonDecay
from .train import fill_buffer, train_with_evaluation

PANEL_SEEDS: list[int] = list(range(10_000, 10_020))


@dataclass
class ExperimentConfig:
    """Hiperparámetros de un entrenamiento; los valores por defecto son los elegidos en el Ej. 2."""

    name: str = "base"
    seed: int = 0
    asset: str = "rep"
    split: str = "train"
    # entorno
    reward_mode: str = "twap"
    state_mode: str = "plan"
    tracking_scale: float = 0.2
    pending_penalty: float = 2.0
    target_horizon: float = 1.0
    dev_band: float = 0.0
    price_weight: float = 0.0
    # agente
    gamma: float = 0.9
    alpha: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 20_000
    buffer_fill: int = 5_000
    hidden_neurons: int = 32
    learn_after: int = 4
    update_target_after: int = 500
    # exploración y entrenamiento
    epsilon: str = "linear"
    eps_min: float = 0.05
    eps_decay: float = 1.0 / 150
    nepisodes: int = 150
    eval_every: int = 5
    panel_seeds: list[int] = field(default_factory=lambda: list(PANEL_SEEDS))
    stop_ratio: float | None = None
    stop_patience: int = 5
    checkpoint_dir: str | None = None
    checkpoint_every: int = 25


def build_env(cfg: ExperimentConfig, split: str | None = None) -> BestExecutionEnv:
    """Crea el entorno de la configuración (por defecto sobre su split)."""
    data = load_market_data(asset=cfg.asset, split=split or cfg.split)
    return BestExecutionEnv(
        data,
        reward_mode=cfg.reward_mode,
        state_mode=cfg.state_mode,
        tracking_scale=cfg.tracking_scale,
        pending_penalty=cfg.pending_penalty,
        target_horizon=cfg.target_horizon,
        dev_band=cfg.dev_band,
        price_weight=cfg.price_weight,
        # el bloque terminal descuenta con el mismo γ que el agente
        completion_discount=cfg.gamma,
    )


def build_agent(cfg: ExperimentConfig, env: BestExecutionEnv) -> DDQNAgent:
    """Crea el DDQN de la configuración, sin compra forzada en el primer step."""
    return DDQNAgent(
        env,
        gamma=cfg.gamma,
        epsilon=1.0,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        hidden_neurons=cfg.hidden_neurons,
        force_first_buy=False,
    )


def build_scheduler(cfg: ExperimentConfig) -> LinearEpsilonDecay | AdaptiveEpsilon:
    """Crea el esquema de ε de la configuración."""
    if cfg.epsilon == "adaptive":
        return AdaptiveEpsilon(eps_max=1.0, eps_min=cfg.eps_min)
    return LinearEpsilonDecay(eps_start=1.0, eps_min=cfg.eps_min, decay=cfg.eps_decay)


def twap_panel(env: BestExecutionEnv, seeds: list[int], horizon: float = 1.0) -> pd.DataFrame:
    """Evalúa el TWAP determinista sobre el panel de episodios."""
    return evaluate_panel(env, TWAP(env, horizon=horizon, force_first_buy=False), seeds)


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """Entrena una configuración de principio a fin y devuelve historial, evaluaciones y pesos."""
    torch.set_num_threads(1)
    set_seeds(cfg.seed)
    env = build_env(cfg)
    agent = build_agent(cfg, env)
    scheduler = build_scheduler(cfg)
    baseline = summarize_panel(twap_panel(env, cfg.panel_seeds, cfg.target_horizon))["mean_abs_dev"]
    fill_buffer(env, agent, buffer_steps=cfg.buffer_fill, log_every=0)
    history, eval_df = train_with_evaluation(
        env,
        agent,
        scheduler,
        panel_seeds=cfg.panel_seeds,
        baseline_error=baseline,
        nepisodes=cfg.nepisodes,
        learn_after=cfg.learn_after,
        update_target_after=cfg.update_target_after,
        eval_every=cfg.eval_every,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_every=cfg.checkpoint_every,
        stop_ratio=cfg.stop_ratio,
        stop_patience=cfg.stop_patience,
        verbose=False,
    )
    eval_df.insert(0, "name", cfg.name)
    eval_df.insert(1, "seed", cfg.seed)
    return {
        "config": asdict(cfg),
        "history": history,
        "eval": eval_df,
        "baseline_error": baseline,
        "weights": {k: v.cpu() for k, v in agent.primary_model.state_dict().items()},
    }


def run_many(configs: list[ExperimentConfig], n_jobs: int = 8) -> list[dict[str, Any]]:
    """Ejecuta varias configuraciones en procesos separados (resultados en el mismo orden)."""
    if n_jobs <= 1:
        return [run_experiment(c) for c in configs]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(run_experiment, configs))


def restore_agent(result: dict[str, Any], split: str | None = None) -> tuple[BestExecutionEnv, DDQNAgent]:
    """Reconstruye entorno y agente entrenado a partir del resultado de run_experiment."""
    cfg = ExperimentConfig(**result["config"])
    env = build_env(cfg, split=split)
    agent = build_agent(cfg, env)
    agent.primary_model.load_state_dict(result["weights"])
    agent.update_target()
    agent.set_trainable(False)
    return env, agent


def with_seeds(cfg: ExperimentConfig, seeds: list[int], **changes: Any) -> list[ExperimentConfig]:
    """Replica una configuración para varias semillas aplicando cambios."""
    return [replace(cfg, seed=s, **changes) for s in seeds]


def eval_curves(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Concatena las evaluaciones periódicas de varios experimentos en formato largo."""
    return pd.concat([r["eval"] for r in results], ignore_index=True)


def save_results_table(df: pd.DataFrame, path: str | Path) -> None:
    """Guarda una tabla de resultados en CSV, creando la carpeta si hace falta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


__all__ = [
    "ExperimentConfig",
    "PANEL_SEEDS",
    "build_env",
    "build_agent",
    "build_scheduler",
    "twap_panel",
    "run_experiment",
    "run_many",
    "restore_agent",
    "with_seeds",
    "eval_curves",
    "save_results_table",
]
