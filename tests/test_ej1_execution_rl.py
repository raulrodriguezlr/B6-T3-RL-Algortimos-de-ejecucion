"""Suite de tests unitarios y de integración para el Ejercicio 1 de ejecución con RL."""

import copy
import datetime
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from execution_rl import (
    TWAP,
    BestExecutionEnv,
    DDQNAgent,
    evaluate_episode,
    fill_buffer,
    load_market_data,
    set_seeds,
    train_agent,
)


@pytest.fixture()
def synthetic_market_data() -> dict[datetime.date, pd.DataFrame]:
    """Construye un diccionario sintético de dos días bursátiles sin acceso a red."""
    rng = np.random.default_rng(12345)
    data: dict[datetime.date, pd.DataFrame] = {}
    n_rows = 800
    for day_idx, day in enumerate([datetime.date(2019, 1, 2), datetime.date(2019, 1, 3)]):
        idx = pd.date_range(
            start=f"{day.isoformat()} 13:00:00",
            periods=n_rows,
            freq="5s",
        )
        base_price = 14.0 + 0.1 * day_idx + np.linspace(0.0, 0.2, n_rows)
        noise = rng.normal(0.0, 0.005, size=n_rows)
        bid1 = np.round(base_price + noise, 4)
        ask1 = np.round(bid1 + 0.01, 4)
        vbid1 = rng.integers(100, 1000, size=n_rows)
        vask1 = rng.integers(100, 1000, size=n_rows)
        step_vol = rng.integers(50, 500, size=n_rows)
        cumvol = np.cumsum(step_vol)
        df = pd.DataFrame(
            {
                "ask1": ask1,
                "bid1": bid1,
                "vask1": vask1,
                "vbid1": vbid1,
                "last": np.round((ask1 + bid1) / 2.0, 4),
                "cumvol": cumvol,
            },
            index=idx,
        )
        df.index.name = "time"
        data[day] = df
    return data


def test_set_seeds_reproducibility() -> None:
    """Verifica que set_seeds sincroniza las secuencias aleatorias de Python, NumPy y PyTorch."""
    set_seeds(99)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(5)
    th_a = torch.randn(5)

    set_seeds(99)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(5)
    th_b = torch.randn(5)

    assert py_a == py_b
    assert np.allclose(np_a, np_b)
    assert torch.allclose(th_a, th_b)


def test_load_market_data_local_and_errors() -> None:
    """Comprueba la carga local desde data/ y el error ante archivos inexistentes."""
    val_data = load_market_data(asset="rep", split="validation", data_dir="data")
    assert len(val_data) == 30
    first_day, first_df = next(iter(val_data.items()))
    assert isinstance(first_day, datetime.date)
    assert list(first_df.columns) == ["ask1", "bid1", "vask1", "vbid1", "last", "cumvol"]
    assert not first_df.isna().any().any()

    with pytest.raises(FileNotFoundError):
        load_market_data(asset="inexistente", split="train", data_dir="data")


def test_block_1_and_2_episode_generation(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica los bloques (1) y (2): rangos de parámetros y dimensiones del episodio."""
    set_seeds(42)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    for _ in range(15):
        env._reset_env_episode_params()
        env._generate_episode()
        assert env.episode_bins is not None
        assert 400 <= env.episode_bins <= 600
        assert isinstance(env.vol_care, int)
        assert int(0.075 * env.episode_bins) <= env.vol_care <= int(0.125 * env.episode_bins)
        assert env.episode_full_len == env.episode_bins + env.look_back
        assert env.episode_full is not None and env.episode is not None
        assert len(env.episode_full) == env.episode_bins + env.look_back
        assert len(env.episode) == env.episode_bins
        assert env.episode_full.index[-env.episode_bins :].equals(env.episode.index)


def test_block_3_observation_builder(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica el bloque (3): observación inicial [1.0, 1.0] y evolución acotada en [0, 1]."""
    set_seeds(42)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    obs0 = env.reset()
    assert obs0.shape == (2,)
    assert np.allclose(obs0, np.array([1.0, 1.0], dtype=np.float32))

    obs1, _, done, _ = env.step(1)
    assert not done
    assert obs1 is not None
    expected_time = (env.episode_bins - 1) / env.episode_bins
    expected_vol = 1.0 - (1 / env.vol_care)
    assert np.allclose(obs1, np.array([expected_time, expected_vol], dtype=np.float32))
    assert np.all((obs1 >= 0.0) & (obs1 <= 1.0))


def test_block_4_and_5_market_vwap_causality_and_stats_df(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica los bloques (4) y (5): causalidad de _compute_market_vwap y coherencia con stats_df."""
    set_seeds(42)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    env.reset()

    env.state_pos = 10
    vwap_before = env._compute_market_vwap()

    # Alterar precios y volúmenes en el futuro (posiciones > 10) no debe afectar al VWAP causal
    env.episode.iloc[11:, env.episode.columns.get_loc("ask1")] = 9999.0
    env.episode.iloc[11:, env.episode.columns.get_loc("bid1")] = 9999.0
    env.market_ep_vol.iloc[11:] = 500000.0
    vwap_after = env._compute_market_vwap()
    assert np.isclose(vwap_before, vwap_after)

    # Coherencia entre market_vwap_hist recolectado paso a paso y market_df de stats_df()
    set_seeds(42)
    env2 = BestExecutionEnv(synthetic_market_data, look_back=60)
    env2.reset()
    for _ in range(25):
        _, _, done, _ = env2.step(0)
        if done:
            break
    _, market_df, _ = env2.stats_df()
    n_recorded = len(env2.market_vwap_hist)
    assert np.allclose(
        env2.market_vwap_hist,
        market_df["vwap"].iloc[:n_recorded].values,
    )


def test_block_6_compute_reward_modes(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica el bloque (6): recompensa nula con vol=0 y referencia global frente a causal."""
    set_seeds(42)
    env_global = BestExecutionEnv(
        synthetic_market_data, look_back=60, use_causal_vwap_reward=False
    )
    env_global.reset()
    env_global.state_pos = 5
    price = 14.05

    assert env_global._compute_reward(price=price, vol=0) == 0.0
    expected_global = (env_global.episode_vwap - price) / env_global.episode_vwap
    assert np.isclose(env_global._compute_reward(price=price, vol=1), expected_global)

    set_seeds(42)
    env_causal = BestExecutionEnv(
        synthetic_market_data, look_back=60, use_causal_vwap_reward=True
    )
    env_causal.reset()
    env_causal.state_pos = 5
    causal_vwap = env_causal._compute_market_vwap()
    expected_causal = (causal_vwap - price) / causal_vwap
    assert np.isclose(env_causal._compute_reward(price=price, vol=1), expected_causal)


def test_block_7_8_9_actions_and_do_nothing_at_start(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica los bloques (7), (8) y (9), incluyendo _do_nothing en state_pos == 0."""
    set_seeds(42)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    env.reset()

    # En state_pos == 0, _do_nothing no debe fallar aunque algo_vwap_hist esté vacío
    r0 = env._do_nothing()
    assert r0 == 0.0
    assert env.exec_vol == 0
    assert env.algo_vwap == 0.0
    assert env.algo_vwap_hist == [0.0]
    assert env.actions_hist == [0]

    env.state_pos = 1
    ask_p1 = float(env.episode["ask1"].values[1])
    r1 = env._agg_action()
    assert env.exec_vol == 1
    assert np.isclose(env.algo_vwap, ask_p1)
    assert np.isclose(r1, (env.episode_vwap - ask_p1) / env.episode_vwap)

    env.state_pos = 2
    r2 = env._do_nothing()
    assert r2 == 0.0
    assert env.exec_vol == 1
    assert np.isclose(env.algo_vwap, ask_p1)


def test_block_10_11_12_stop_conditions_and_done_reward(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica los bloques (10), (11) y (12): parada por completar orden frente a agotar tiempo."""
    set_seeds(42)
    env = BestExecutionEnv(synthetic_market_data, look_back=60, done_penalty=-2.5)
    env.reset()

    # Caso 1: agotar el tiempo sin completar vol_care
    done = False
    last_reward = 0.0
    while not done:
        obs, last_reward, done, _ = env.step(0)
    assert obs is None
    assert env.state_pos == env.episode_bins
    assert env.exec_vol == 0
    assert np.isclose(last_reward, -2.5)

    # Caso 2: completar exactamente vol_care antes de agotar episode_bins
    env.reset()
    target_vol = env.vol_care
    for _ in range(target_vol):
        obs, last_reward, done, _ = env.step(1)
    assert done
    assert obs is None
    assert env.exec_vol == target_vol
    assert env.state_pos == target_vol
    expected_step_rew = (
        env.episode_vwap - float(env.episode["ask1"].values[target_vol - 1])
    ) / env.episode_vwap
    assert np.isclose(last_reward, expected_step_rew)


def test_block_13_action_sample_and_step_validation(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica el bloque (13) y la validación de acciones inválidas en step."""
    set_seeds(42)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    env.reset()

    samples = [env.action_sample() for _ in range(2000)]
    assert set(samples).issubset({0, 1})
    expected_p = env.vol_care / env.episode_bins
    assert abs(np.mean(samples) - expected_p) < 0.03

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(2)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(-1)


def test_ddqn_agent_network_cer_and_learn(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica la arquitectura PyTorch, muestreo CER, modos de act y paso de aprendizaje DDQN."""
    set_seeds(42)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    s0 = env.reset()
    agent = DDQNAgent(env, batch_size=8, buffer_size=100, hidden_neurons=16)

    # Pesos iniciales de primary y target sincronizados
    for p_prim, p_targ in zip(
        agent.primary_model.parameters(), agent.target_model.parameters()
    ):
        assert torch.allclose(p_prim, p_targ)

    # getQs y getQ
    qs = agent.getQs(s0, agent.primary_model)
    assert qs.shape == (1, 2)
    q0 = agent.getQ(s0, 0)
    assert np.isclose(q0[0], qs[0, 0])
    assert agent.getQs(np.array([]), agent.primary_model).size == 0

    # Comportamiento de force_first_buy en state_pos == 0 y modos trainable/eval
    env.state_pos = 0
    agent.set_trainable(False)
    agent.force_first_buy = True
    assert agent.act(s0) == 1

    agent.force_first_buy = False
    greedy_action = int(np.argmax(qs))
    assert agent.act(s0) == greedy_action

    agent.set_trainable(True)
    agent.epsilon = 0.0
    assert agent.act(s0) == greedy_action
    agent.force_first_buy = True

    # Buffer insuficiente debe lanzar ValueError
    agent.experience(s0, 1, 0.1, s0, done=False)
    with pytest.raises(ValueError, match="Buffer insuficiente"):
        agent.sample()

    # Rellenar hasta batch_size y verificar CER (última transición siempre incluida)
    for i in range(1, 10):
        s_next = np.array([1.0 - 0.01 * i, 1.0 - 0.01 * i], dtype=np.float32)
        agent.experience(s_next, i % 2, float(i), s_next, done=(i == 9))

    last_exp = agent.buffer[-1]
    assert last_exp[3] is None
    for _ in range(10):
        batch = agent.sample()
        assert len(batch) == agent.batch_size
        assert batch[-1][2] == last_exp[2] and batch[-1][3] is None

    # Learn actualiza primary_model pero mantiene intacto target_model hasta update_target()
    prim_before = [p.detach().clone() for p in agent.primary_model.parameters()]
    targ_before = [p.detach().clone() for p in agent.target_model.parameters()]

    loss = agent.learn()
    assert isinstance(loss, float) and np.isfinite(loss)

    prim_changed = any(
        not torch.allclose(p_old, p_new)
        for p_old, p_new in zip(prim_before, agent.primary_model.parameters())
    )
    targ_unchanged = all(
        torch.allclose(t_old, t_new)
        for t_old, t_new in zip(targ_before, agent.target_model.parameters())
    )
    assert prim_changed
    assert targ_unchanged

    agent.update_target()
    for p_prim, p_targ in zip(
        agent.primary_model.parameters(), agent.target_model.parameters()
    ):
        assert torch.allclose(p_prim, p_targ)


def test_twap_agent_completes_order_deterministically(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica que el agente TWAP completa el 100% de vol_care de forma determinista."""
    set_seeds(42)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    twap = TWAP(env)
    metrics = evaluate_episode(env, twap, seed=42)
    assert metrics["exec_vol"] == metrics["vol_care"]
    assert metrics["steps"] <= env.episode_bins


def test_training_loop_fill_buffer_and_reproducibility(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica fill_buffer, train_agent, evaluate_episode en otro entorno y reproducibilidad."""
    set_seeds(77)
    env_a = BestExecutionEnv(synthetic_market_data, look_back=60)
    agent_a = DDQNAgent(
        env_a, epsilon=0.8, batch_size=16, buffer_size=200, hidden_neurons=16
    )
    fill_buffer(env_a, agent_a, buffer_steps=60, log_every=0)
    assert len(agent_a.buffer) == 60
    assert np.isclose(agent_a.epsilon, 0.8)

    hist_a = train_agent(
        env_a,
        agent_a,
        nepisodes=4,
        min_epsilon=0.1,
        epsilon_decay=0.1,
        learn_after=16,
        update_target_after=32,
        n_log=0,
        verbose=False,
    )
    assert len(hist_a["rewards"]) == 4
    assert len(hist_a["losses"]) == 4
    assert np.isclose(agent_a.epsilon, 0.4)

    # Reproducibilidad exacta con la misma semilla
    set_seeds(77)
    env_b = BestExecutionEnv(copy.deepcopy(synthetic_market_data), look_back=60)
    agent_b = DDQNAgent(
        env_b, epsilon=0.8, batch_size=16, buffer_size=200, hidden_neurons=16
    )
    fill_buffer(env_b, agent_b, buffer_steps=60, log_every=0)
    hist_b = train_agent(
        env_b,
        agent_b,
        nepisodes=4,
        min_epsilon=0.1,
        epsilon_decay=0.1,
        learn_after=16,
        update_target_after=32,
        n_log=0,
        verbose=False,
    )
    assert np.allclose(hist_a["rewards"], hist_b["rewards"])
    assert np.allclose(hist_a["losses"], hist_b["losses"])

    # Evaluar en un segundo entorno verifica que agent.env se sincroniza y restaura
    env_val = BestExecutionEnv(synthetic_market_data, look_back=60)
    agent_a.set_trainable(True)
    eval_metrics_1 = evaluate_episode(env_val, agent_a, seed=101)
    eval_metrics_2 = evaluate_episode(env_val, agent_a, seed=101)
    assert agent_a.env is env_a
    assert agent_a._trainable is True
    assert eval_metrics_1 == eval_metrics_2


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_vwap_convention_end_matches_episode_vwap(
    synthetic_market_data: dict[datetime.date, pd.DataFrame], seed: int
) -> None:
    """Verifica que el VWAP de mercado final coincide con episode_vwap en stats_df y en el cálculo causal."""
    set_seeds(seed)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(0)
    _, market_df, _ = env.stats_df()
    assert abs(float(market_df["vwap"].iloc[-1]) - env.episode_vwap) < 1e-9

    env.state_pos = env.episode_bins - 1
    assert abs(env._compute_market_vwap() - env.episode_vwap) < 1e-9


@pytest.mark.parametrize("seed", [0, 5, 42])
def test_compute_market_vwap_causal_at_several_positions(
    synthetic_market_data: dict[datetime.date, pd.DataFrame], seed: int
) -> None:
    """Verifica que alterar datos posteriores a t no cambia _compute_market_vwap en t."""
    set_seeds(seed)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    env.reset()
    for t in (1, 50, env.episode_bins // 2):
        env.state_pos = t
        before = env._compute_market_vwap()
        ep_orig = env.episode
        vol_orig = env.market_ep_vol
        env.episode = ep_orig.copy()
        env.market_ep_vol = vol_orig.copy()
        env.episode.iloc[t + 1 :, env.episode.columns.get_loc("ask1")] = 1e6
        env.episode.iloc[t + 1 :, env.episode.columns.get_loc("bid1")] = 1e6
        env.market_ep_vol.iloc[t + 1 :] = 1e9
        after = env._compute_market_vwap()
        env.episode = ep_orig
        env.market_ep_vol = vol_orig
        assert before == pytest.approx(after, abs=1e-12)


def test_market_ep_vol_clipped_with_decreasing_cumvol(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica que el volumen de mercado nunca es negativo aunque cumvol baje."""
    data = copy.deepcopy(synthetic_market_data)
    for df in data.values():
        cv = df["cumvol"].to_numpy().astype(float).copy()
        for k in range(30, len(cv), 40):
            cv[k:] -= 5000.0
        df["cumvol"] = cv
    set_seeds(42)
    env = BestExecutionEnv(data, look_back=60)
    for _ in range(5):
        env.reset()
        assert (env.episode["cumvol"].diff().iloc[1:] < 0).any()
        assert (env.market_ep_vol >= 0).all()
        assert env.market_ep_vol.iloc[0] == 0.0
        assert np.isfinite(env.episode_vwap)


def test_evaluate_episode_avg_half_spread_bps(
    synthetic_market_data: dict[datetime.date, pd.DataFrame],
) -> None:
    """Verifica que evaluate_episode devuelve avg_half_spread_bps finito, positivo y correcto."""
    set_seeds(42)
    env = BestExecutionEnv(synthetic_market_data, look_back=60)
    metrics = evaluate_episode(env, TWAP(env), seed=42)
    assert "avg_half_spread_bps" in metrics
    val = metrics["avg_half_spread_bps"]
    assert np.isfinite(val) and val > 0.0
    ep = env.episode
    mid = (ep["ask1"] + ep["bid1"]) / 2.0
    expected = float((((ep["ask1"] - ep["bid1"]) / 2.0) / mid).mean() * 1e4)
    assert val == pytest.approx(expected, rel=1e-9)
