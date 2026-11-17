"""Funciones de recolección de experiencias, entrenamiento y evaluación de agentes."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .agents import DDQNAgent
from .data import set_seeds
from .env import BestExecutionEnv
from .evaluation import evaluate_panel, summarize_panel


def fill_buffer(
    env: BestExecutionEnv,
    agent: DDQNAgent,
    buffer_steps: int | None = None,
    log_every: int = 10000,
) -> None:
    """Rellena el buffer con experiencias exploratorias forzando temporalmente épsilon igual a 1."""
    steps = int(buffer_steps if buffer_steps is not None else agent.buffer_size)
    prev_epsilon = agent.epsilon
    prev_trainable = agent._trainable

    agent.env = env
    agent.epsilon = 1.0
    agent.set_trainable(True)
    try:
        s = env.reset()
        for exps in range(steps):
            a = agent.act(s)
            s1, r, done, _ = env.step(a)
            agent.experience(s, a, r, s1, done)
            s = s1
            if log_every and exps % log_every == 0:
                print(f"buffer exps: {exps}")
            if done:
                s = env.reset()
    finally:
        agent.epsilon = prev_epsilon
        agent.set_trainable(prev_trainable)


def train_agent(
    env: BestExecutionEnv,
    agent: DDQNAgent,
    nepisodes: int = 200,
    min_epsilon: float = 0.05,
    epsilon_decay: float = 0.005,
    learn_after: int | None = None,
    update_target_after: int | None = None,
    n_log: int = 25,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Ejecuta el bucle de entrenamiento de Double DQN respetando la secuencia de interacción."""
    learn_step = int(learn_after if learn_after is not None else agent.batch_size)
    target_step = int(
        update_target_after if update_target_after is not None else 10 * learn_step
    )

    agent.env = env
    agent.set_trainable(True)
    learn_counter = 0
    mse = float("nan")
    history_steps: list[float] = []
    history_rewards: list[float] = []
    history_disc_rewards: list[float] = []
    history_losses: list[float] = []
    history_epsilons: list[float] = []

    for episode in range(nepisodes):
        s = env.reset()
        step = 0
        cum_reward = 0.0
        dis_cum_reward = 0.0
        while True:
            a = agent.act(s)
            s1, r, done, _ = env.step(a)
            agent.experience(s, a, r, s1, done)
            learn_counter += 1
            cum_reward += r
            dis_cum_reward += (agent.gamma**step) * r
            s = s1
            step += 1

            if learn_counter % learn_step == 0 and len(agent.buffer) >= agent.batch_size:
                mse = agent.learn()
            if learn_counter % target_step == 0:
                agent.update_target()

            if done:
                agent.epsilon = max(agent.epsilon - epsilon_decay, min_epsilon)
                history_rewards.append(float(cum_reward))
                history_disc_rewards.append(float(dis_cum_reward))
                history_losses.append(float(mse))
                history_steps.append(float(step))
                history_epsilons.append(float(agent.epsilon))

                if verbose and n_log and episode % n_log == 0:
                    mse_str = (
                        f"{np.round(mse, 6)}" if not np.isnan(mse) else "nan"
                    )
                    print(
                        f"Episode: {episode}, "
                        f"steps: {np.round(np.mean(history_steps[-n_log:]), 2)}, "
                        f"rew: {np.round(np.mean(history_rewards[-n_log:]), 4)}, "
                        f"mse: {mse_str}, "
                        f"eps: {np.round(agent.epsilon, 2)}"
                    )
                break

    return {
        "steps": history_steps,
        "rewards": history_rewards,
        "disc_rewards": history_disc_rewards,
        "losses": history_losses,
        "epsilons": history_epsilons,
    }


def evaluate_episode(
    env: BestExecutionEnv,
    agent: DDQNAgent,
    seed: int | None = None,
) -> dict[str, float]:
    """Evalúa un episodio en modo determinista sin exploración y devuelve métricas de ejecución."""
    if seed is not None:
        set_seeds(seed)

    prev_env = agent.env
    prev_trainable = agent._trainable
    agent.env = env
    agent.set_trainable(False)
    try:
        s = env.reset()
        step = 0
        cum_reward = 0.0
        disc_cum_reward = 0.0
        while True:
            a = agent.act(s)
            s, r, done, _ = env.step(a)
            cum_reward += r
            disc_cum_reward += (agent.gamma**step) * r
            step += 1
            if done:
                break
    finally:
        agent.env = prev_env
        agent.set_trainable(prev_trainable)

    algo_vwap = float(env.algo_vwap_hist[-1]) if env.algo_vwap_hist else 0.0
    market_vwap = float(env.episode_vwap) if env.episode_vwap is not None else 0.0
    if market_vwap > 0.0 and algo_vwap > 0.0:
        vwap_slippage_bps = float(((market_vwap - algo_vwap) / market_vwap) * 10000.0)
    else:
        vwap_slippage_bps = 0.0

    ep = env.episode
    mid = (ep["ask1"] + ep["bid1"]) / 2.0
    avg_half_spread_bps = float((((ep["ask1"] - ep["bid1"]) / 2.0) / mid * 1e4).mean())

    return {
        "avg_half_spread_bps": avg_half_spread_bps,
        "steps": float(step),
        "exec_vol": float(env.exec_vol),
        "vol_care": float(env.vol_care),
        "cum_reward": float(cum_reward),
        "disc_cum_reward": float(disc_cum_reward),
        "algo_vwap": algo_vwap,
        "market_vwap": market_vwap,
        "vwap_slippage_bps": vwap_slippage_bps,
    }


def save_checkpoint(
    path: str | Path,
    agent: DDQNAgent,
    scheduler: Any,
    episode: int,
    history: dict[str, list[float]],
    eval_rows: list[dict[str, float]],
    save_buffer: bool = True,
    streak: int = 0,
) -> None:
    """Guarda pesos, optimizador, estado de ε, historial y (opcional) buffer para poder reanudar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "primary": agent.primary_model.state_dict(),
            "target": agent.target_model.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "episode": episode,
            "history": history,
            "eval_rows": eval_rows,
            "streak": streak,
            "buffer": list(agent.buffer) if save_buffer else None,
            "rng": {"numpy": np.random.get_state(), "torch": torch.get_rng_state()},
        },
        path,
    )


def load_checkpoint(path: str | Path, agent: DDQNAgent, scheduler: Any | None = None) -> dict[str, Any]:
    """Restaura pesos, optimizador, ε, buffer y RNG en el agente; devuelve el checkpoint completo.

    Usa pickle (weights_only=False): solo para checkpoints generados por este proyecto.
    """
    ckpt = torch.load(Path(path), weights_only=False)
    agent.primary_model.load_state_dict(ckpt["primary"])
    agent.target_model.load_state_dict(ckpt["target"])
    agent.optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
        agent.epsilon = scheduler.epsilon
    if ckpt.get("buffer") is not None:
        agent.buffer.clear()
        agent.buffer.extend(ckpt["buffer"])
    np.random.set_state(ckpt["rng"]["numpy"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    return ckpt


def train_with_evaluation(
    env: BestExecutionEnv,
    agent: DDQNAgent,
    scheduler: Any,
    panel_seeds: list[int],
    baseline_error: float,
    nepisodes: int = 200,
    learn_after: int = 4,
    update_target_after: int = 500,
    eval_every: int = 5,
    eval_env: BestExecutionEnv | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 25,
    stop_ratio: float | None = None,
    stop_patience: int = 5,
    resume_from: str | Path | None = None,
    n_log: int = 25,
    verbose: bool = True,
) -> tuple[dict[str, list[float]], pd.DataFrame]:
    """Bucle DDQN con evaluación greedy periódica sobre un panel fijo, ε del scheduler y checkpoints.

    Parada (Ej. 4C): ratio <= stop_ratio durante stop_patience evaluaciones seguidas con ε en su suelo.
    """
    eval_env = eval_env if eval_env is not None else env
    history: dict[str, list[float]] = {k: [] for k in ("steps", "rewards", "losses", "epsilons")}
    eval_rows: list[dict[str, float]] = []
    start_episode = 0
    learn_counter = 0
    if resume_from is not None:
        ckpt = load_checkpoint(resume_from, agent, scheduler)
        history = ckpt["history"]
        eval_rows = ckpt["eval_rows"]
        start_episode = int(ckpt["episode"]) + 1
        learn_counter = int(sum(history["steps"]))
        streak_resume = int(ckpt.get("streak", 0))
    else:
        streak_resume = 0

    agent.env = env
    agent.set_trainable(True)
    agent.epsilon = scheduler.epsilon
    mse = float("nan")
    best_ratio = min((r["ratio"] for r in eval_rows), default=np.inf)
    streak = streak_resume
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    def _evaluate(episode: int) -> float:
        """Evalúa la política greedy y devuelve el ratio de error frente al TWAP."""
        summary = summarize_panel(evaluate_panel(eval_env, agent, panel_seeds))
        ratio = summary["mean_abs_dev"] / baseline_error
        eps = scheduler.on_evaluation(ratio)
        eval_rows.append({"episode": float(episode), "ratio": ratio, "epsilon": eps, **summary})
        return ratio

    if start_episode == 0:
        _evaluate(-1)  # punto de partida: red sin entrenar
        agent.epsilon = scheduler.epsilon

    for episode in range(start_episode, nepisodes):
        s = env.reset()
        step, cum_reward = 0, 0.0
        while True:
            a = agent.act(s)
            s1, r, done, _ = env.step(a)
            agent.experience(s, a, r, s1, done)
            learn_counter += 1
            cum_reward += r
            s = s1
            step += 1
            if learn_counter % learn_after == 0 and len(agent.buffer) >= agent.batch_size:
                mse = agent.learn()
            if learn_counter % update_target_after == 0:
                agent.update_target()
            if done:
                break

        agent.epsilon = scheduler.on_episode_end()
        history["steps"].append(float(step))
        history["rewards"].append(float(cum_reward))
        history["losses"].append(float(mse))
        history["epsilons"].append(float(agent.epsilon))

        stop = False
        if (episode + 1) % eval_every == 0:
            ratio = _evaluate(episode)
            agent.epsilon = scheduler.epsilon
            if ratio < best_ratio:
                best_ratio = ratio
                if ckpt_dir is not None:
                    save_checkpoint(ckpt_dir / "best.pt", agent, scheduler, episode, history, eval_rows, save_buffer=False)
            if stop_ratio is not None:
                at_floor = scheduler.epsilon <= scheduler.eps_min + 0.01
                streak = streak + 1 if (ratio <= stop_ratio and at_floor) else 0
                stop = streak >= stop_patience

        if ckpt_dir is not None and ((episode + 1) % checkpoint_every == 0 or stop):
            save_checkpoint(ckpt_dir / "last.pt", agent, scheduler, episode, history, eval_rows, streak=streak)

        if verbose and n_log and (episode % n_log == 0 or stop):
            last = eval_rows[-1]
            print(
                f"Episode: {episode}, steps: {step}, rew: {np.round(cum_reward, 2)}, "
                f"mse: {np.round(mse, 4)}, eps: {np.round(agent.epsilon, 3)}, "
                f"ratio vs TWAP: {np.round(last['ratio'], 3)}"
            )
        if stop:
            if verbose:
                print(f"Parada por aprendizaje en el episodio {episode}: ratio <= {stop_ratio} en {stop_patience} evaluaciones seguidas")
            break

    return history, pd.DataFrame(eval_rows)
