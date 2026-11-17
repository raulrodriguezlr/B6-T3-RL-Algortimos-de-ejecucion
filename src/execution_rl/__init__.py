"""Paquete de aprendizaje por refuerzo aplicado a algoritmos de ejecución."""

from .agents import DDQNAgent, TWAP
from .data import load_market_data, set_seeds
from .env import BestExecutionEnv
from .evaluation import episode_trace, evaluate_panel, run_episode, summarize_panel
from .exploration import AdaptiveEpsilon, LinearEpsilonDecay
from .report import plot_eval_curves, plot_results, plot_traces_vs_plan, plot_training_rewards
from .train import (
    evaluate_episode,
    fill_buffer,
    load_checkpoint,
    save_checkpoint,
    train_agent,
    train_with_evaluation,
)

__all__ = [
    "load_market_data",
    "set_seeds",
    "BestExecutionEnv",
    "DDQNAgent",
    "TWAP",
    "fill_buffer",
    "train_agent",
    "evaluate_episode",
    "plot_results",
    "plot_training_rewards",
    "run_episode",
    "evaluate_panel",
    "summarize_panel",
    "LinearEpsilonDecay",
    "AdaptiveEpsilon",
    "train_with_evaluation",
    "save_checkpoint",
    "load_checkpoint",
    "episode_trace",
    "plot_traces_vs_plan",
    "plot_eval_curves",
]
