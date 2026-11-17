"""Funciones de recolección de experiencias, entrenamiento y evaluación de agentes."""

import numpy as np

from .agents import DDQNAgent
from .data import set_seeds
from .env import BestExecutionEnv


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
