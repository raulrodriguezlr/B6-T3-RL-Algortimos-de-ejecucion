"""Tests de recompensas TWAP, estados ampliados, exploración adaptativa, evaluación y entrenamiento (Ej. 2 a 4)."""

import copy
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from execution_rl import (
    TWAP,
    AdaptiveEpsilon,
    BestExecutionEnv,
    DDQNAgent,
    LinearEpsilonDecay,
    episode_trace,
    evaluate_panel,
    fill_buffer,
    run_episode,
    set_seeds,
    summarize_panel,
    train_with_evaluation,
)
from execution_rl.experiments import (
    ExperimentConfig,
    build_scheduler,
    run_many,
    with_seeds,
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


def make_env(data: dict[datetime.date, pd.DataFrame], **kwargs) -> BestExecutionEnv:
    """Crea un entorno pequeño sobre datos sintéticos."""
    params = dict(look_back=20, min_bins=100, max_bins=120)
    params.update(kwargs)
    return BestExecutionEnv(data, **params)


def make_agent(env: BestExecutionEnv, **kwargs) -> DDQNAgent:
    """Crea un DDQN diminuto sin compra forzada."""
    params = dict(batch_size=8, buffer_size=500, hidden_neurons=8, epsilon=1.0, force_first_buy=False)
    params.update(kwargs)
    return DDQNAgent(env, **params)


# ---------------------------------------------------------------- recompensa twap


def test_twap_step_reward_is_minus_scale_times_deviation(synthetic_market_data) -> None:
    """Recompensa por step = -tracking_scale * |exec_vol - plan(state_pos+1)|."""
    env = make_env(synthetic_market_data, reward_mode="twap", tracking_scale=0.3)
    env.reset(seed=1)
    for action in [0, 0, 1, 0, 1, 1]:
        k = env.state_pos
        exec_after = env.exec_vol + action
        expected = -0.3 * abs(exec_after - env._plan_volume(k + 1))
        _, r, done, _ = env.step(action)
        assert not done
        assert r == pytest.approx(expected)


def test_twap_dev_band_gives_free_margin(synthetic_market_data) -> None:
    """Con dev_band el coste es max(0, |dev| - band)."""
    env = make_env(synthetic_market_data, reward_mode="twap", tracking_scale=0.5, dev_band=2.0)
    env.reset(seed=2)
    for action in [1, 1, 1, 1, 0, 0, 0]:
        k = env.state_pos
        dev = env.exec_vol + action - env._plan_volume(k + 1)
        expected = -0.5 * max(0.0, abs(dev) - 2.0)
        _, r, done, _ = env.step(action)
        assert not done
        assert r == pytest.approx(expected)


def test_twap_price_weight_only_when_buying(synthetic_market_data) -> None:
    """price_weight suma w*(vwap - ask)/vwap*1e4 solo en los steps de compra."""
    env0 = make_env(synthetic_market_data, reward_mode="twap", price_weight=0.0)
    envw = make_env(synthetic_market_data, reward_mode="twap", price_weight=3.0)
    env0.reset(seed=4)
    envw.reset(seed=4)
    for action in [0, 1, 0, 1]:
        ask = float(envw._ask[envw.state_pos])
        _, r0, _, _ = env0.step(action)
        _, rw, _, _ = envw.step(action)
        if action == 1:
            extra = 3.0 * (envw.episode_vwap - ask) / envw.episode_vwap * 1e4
            assert rw - r0 == pytest.approx(extra)
        else:
            assert rw == pytest.approx(r0)


def test_done_reward_pending_at_timeout(synthetic_market_data) -> None:
    """Al agotar el tiempo con orden pendiente: -kappa*lambda*pendiente."""
    env = make_env(synthetic_market_data, reward_mode="twap", tracking_scale=0.2, pending_penalty=3.0)
    env.reset(seed=5)
    env.state_pos = env.episode_bins
    env.exec_vol = env.vol_care - 4
    assert env._compute_done_reward() == pytest.approx(-0.2 * 3.0 * 4)


def test_done_reward_early_completion_discounted(synthetic_market_data) -> None:
    """Orden completa antes de tiempo: -kappa * sum_k gamma^k (V - plan(t+k)), comprobada a mano."""
    env = make_env(
        synthetic_market_data,
        reward_mode="twap",
        tracking_scale=0.2,
        completion_discount=0.8,
        target_horizon=0.9,
    )
    env.reset(seed=6)
    t = env.episode_bins - 30
    env.state_pos = t
    env.exec_vol = env.vol_care
    expected = 0.0
    for k in range(1, env.episode_bins - t + 1):
        plan = env.vol_care * min(1.0, (t + k) / (0.9 * env.episode_bins))
        expected += 0.8**k * (env.vol_care - plan)
    assert env._compute_done_reward() == pytest.approx(-0.2 * expected)
    assert env._compute_done_reward() <= 0.0


def test_done_reward_zero_when_completed_at_end(synthetic_market_data) -> None:
    """Completar justo al final no deja steps restantes: sin término terminal."""
    env = make_env(synthetic_market_data, reward_mode="twap")
    env.reset(seed=7)
    env.state_pos = env.episode_bins
    env.exec_vol = env.vol_care
    assert env._compute_done_reward() == pytest.approx(0.0)


# ---------------------------------------------------------------- plan y validaciones


def test_plan_volume_target_horizon(synthetic_market_data) -> None:
    """El plan alcanza V en 0.9T y se queda plano."""
    env = make_env(synthetic_market_data, target_horizon=0.9)
    env.reset(seed=8)
    T, V = env.episode_bins, env.vol_care
    assert env._plan_volume(0) == 0.0
    assert env._plan_volume(0.45 * T) == pytest.approx(V / 2)
    assert env._plan_volume(0.9 * T) == pytest.approx(V)
    assert env._plan_volume(int(np.ceil(0.9 * T))) == pytest.approx(V)
    assert env._plan_volume(T) == pytest.approx(V)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward_mode": "otro"},
        {"state_mode": "otro"},
        {"target_horizon": 0.0},
        {"target_horizon": 1.5},
        {"target_horizon": -0.1},
    ],
)
def test_env_validations(synthetic_market_data, kwargs) -> None:
    """Parámetros inválidos lanzan ValueError."""
    with pytest.raises(ValueError):
        make_env(synthetic_market_data, **kwargs)


def test_target_horizon_one_is_valid(synthetic_market_data) -> None:
    """El límite superior (0, 1] es inclusivo."""
    make_env(synthetic_market_data, target_horizon=1.0)


# ---------------------------------------------------------------- estados


def test_state_modes_dimensions(synthetic_market_data) -> None:
    """basic=2, plan=3, market=6 y todas las features finitas durante un episodio."""
    for mode, dim in [("basic", 2), ("plan", 3), ("market", 6)]:
        env = make_env(synthetic_market_data, state_mode=mode)
        assert env.n_features == dim
        s = env.reset(seed=9)
        assert s.shape == (dim,)
        rng = np.random.RandomState(0)
        while s is not None:
            assert np.all(np.isfinite(s))
            s, _, done, _ = env.step(int(rng.rand() < 0.3))
            if done:
                break


def test_plan_state_deviation_feature(synthetic_market_data) -> None:
    """dev_feat = clip((exec - plan(state_pos)) / dev_clip, -1, 1) y está en [-1, 1]."""
    env = make_env(synthetic_market_data, state_mode="plan", dev_clip=2.0)
    s = env.reset(seed=10)
    assert s[2] == pytest.approx(0.0)
    for action in [1, 1, 0, 1, 0, 0]:
        s, _, done, _ = env.step(action)
        assert not done
        expected = np.clip((env.exec_vol - env._plan_volume(env.state_pos)) / 2.0, -1.0, 1.0)
        assert s[2] == pytest.approx(expected, abs=1e-6)
        assert -1.0 <= s[2] <= 1.0
    env.exec_vol = 1000
    assert env.observation_builder()[2] == pytest.approx(1.0)
    env.exec_vol = 0
    env.state_pos = 50
    assert env.observation_builder()[2] == pytest.approx(-1.0)


def test_market_features_do_not_use_future(synthetic_market_data) -> None:
    """Alterar las filas posteriores al bin actual no cambia la observación actual."""
    actions = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    env_a = make_env(synthetic_market_data, state_mode="market")
    env_a.reset(seed=11)
    obs_a = []
    for a in actions:
        s, _, done, _ = env_a.step(a)
        assert not done
        obs_a.append(s)
    k = len(actions)
    day = env_a.episode.index[0].date()
    cut = env_a.episode.index[k]

    modified = copy.deepcopy(synthetic_market_data)
    df = modified[day] = modified[day].astype(float)
    future = df.index > cut
    for col in ["ask1", "bid1", "vask1", "vbid1"]:
        df.loc[future, col] = df.loc[future, col] * 1.7 + 3.0
    env_b = make_env(modified, state_mode="market")
    env_b.reset(seed=11)
    assert env_b.episode.index[0] == env_a.episode.index[0]
    obs_b = []
    for a in actions:
        s, _, _, _ = env_b.step(a)
        obs_b.append(s)
    for sa, sb in zip(obs_a, obs_b):
        assert np.allclose(sa, sb)


# ---------------------------------------------------------------- reset(seed)


def test_reset_seed_reproducible_and_global_untouched(synthetic_market_data) -> None:
    """Mismo seed, mismo episodio; el RNG global no se altera."""
    env = make_env(synthetic_market_data)
    np.random.seed(77)
    before = np.random.get_state()
    env.reset(seed=3)
    first = (env.episode.index[0], env.episode_bins, env.vol_care)
    env.reset(seed=3)
    second = (env.episode.index[0], env.episode_bins, env.vol_care)
    after = np.random.get_state()
    assert first == second
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert env._rng is np.random


def test_reset_seed_varies_across_seeds(synthetic_market_data) -> None:
    """Seeds distintos producen episodios distintos (al menos alguno)."""
    env = make_env(synthetic_market_data)
    eps = set()
    for sd in range(8):
        env.reset(seed=sd)
        eps.add((env.episode.index[0], env.episode_bins, env.vol_care))
    assert len(eps) > 1


def test_reset_without_seed_uses_global_rng(synthetic_market_data) -> None:
    """Sin seed, set_seeds reproduce el episodio."""
    env = make_env(synthetic_market_data)
    set_seeds(5)
    env.reset()
    a = (env.episode.index[0], env.episode_bins, env.vol_care)
    set_seeds(5)
    env.reset()
    b = (env.episode.index[0], env.episode_bins, env.vol_care)
    assert a == b


def test_default_vwap_mode_unchanged(synthetic_market_data) -> None:
    """El modo por defecto conserva la recompensa del Ej. 1."""
    env = make_env(synthetic_market_data)
    assert env.reward_mode == "vwap"
    assert env.n_features == 2
    env.reset(seed=12)
    _, r0, _, _ = env.step(0)
    assert r0 == 0.0
    ask = float(env._ask[env.state_pos])
    _, r1, _, _ = env.step(1)
    assert r1 == pytest.approx((env.episode_vwap - ask) / env.episode_vwap)
    env.state_pos = env.episode_bins
    env.exec_vol = env.vol_care - 1
    assert env._compute_done_reward() == env.done_penalty
    env.exec_vol = env.vol_care
    assert env._compute_done_reward() == 0.0


# ---------------------------------------------------------------- TWAP


def test_twap_horizon_one_completes_order(synthetic_market_data) -> None:
    """TWAP(horizon=1) completa la orden en varios episodios."""
    env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan")
    agent = TWAP(env, horizon=1.0, batch_size=8, hidden_neurons=4)
    for sd in range(6):
        m = run_episode(env, agent, seed=sd)
        assert m["completed_pct"] == pytest.approx(100.0)
        assert m["mean_abs_dev"] <= 1.5


def test_twap_horizon_09_finishes_early(synthetic_market_data) -> None:
    """TWAP(horizon=0.9) completa la orden hacia el 90 % del tiempo."""
    env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan", target_horizon=0.9)
    agent = TWAP(env, horizon=0.9, batch_size=8, hidden_neurons=4)
    for sd in range(6):
        m = run_episode(env, agent, seed=sd)
        assert m["completed_pct"] == pytest.approx(100.0)
        assert m["time_used_pct"] <= 90.0 + 5.0
        assert m["time_used_pct"] >= 80.0


# ---------------------------------------------------------------- evaluación


class _NeverBuy:
    """Política trivial que no compra nunca (orden sin ejecutar)."""

    def __init__(self, env: BestExecutionEnv) -> None:
        self.env = env
        self._trainable = False

    def set_trainable(self, trainable: bool) -> None:
        self._trainable = trainable

    def act(self, s: np.ndarray) -> int:
        return 0


def test_cleanup_slippage_equals_slippage_when_complete(synthetic_market_data) -> None:
    """Con la orden completa, el barrido final no compra nada y ambos slippages coinciden."""
    env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan")
    agent = TWAP(env, horizon=1.0, batch_size=8, hidden_neurons=4)
    m = run_episode(env, agent, seed=7)
    assert m["completed_pct"] == pytest.approx(100.0)
    assert m["cleanup_slippage_bps"] == pytest.approx(m["slippage_bps"])


def test_cleanup_slippage_charges_pending_at_last_ask(synthetic_market_data) -> None:
    """Sin compras, todo se compra al ask1 del último bin y el slippage normal no está definido."""
    env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan")
    m = run_episode(env, _NeverBuy(env), seed=7)
    assert m["exec_vol"] == 0.0
    assert np.isnan(m["slippage_bps"])
    last_ask = float(env.episode["ask1"].iloc[-1])
    expected = (env.episode_vwap - last_ask) / env.episode_vwap * 1e4
    assert m["cleanup_slippage_bps"] == pytest.approx(expected)


PANEL_COLUMNS = [
    "seed", "day", "start", "episode_bins", "vol_care", "exec_vol", "completed_pct",
    "time_used_pct", "mean_abs_dev", "max_abs_dev", "off_plan_steps_pct", "cum_reward",
    "slippage_bps", "cleanup_slippage_bps", "half_spread_bps",
]


def test_evaluate_panel_columns_and_restores_agent(synthetic_market_data) -> None:
    """El panel devuelve las columnas esperadas y deja agente y flags como estaban."""
    env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan")
    other_env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan")
    agent = make_agent(env)
    agent.set_trainable(True)
    agent.env = other_env
    df = evaluate_panel(env, agent, [1, 2, 3])
    assert list(df.columns) == PANEL_COLUMNS
    assert len(df) == 3
    assert agent.env is other_env
    assert agent._trainable is True
    agent.set_trainable(False)
    evaluate_panel(env, agent, [1])
    assert agent._trainable is False


def test_evaluate_panel_deterministic(synthetic_market_data) -> None:
    """Mismo panel y política greedy dan el mismo resultado."""
    env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan")
    agent = make_agent(env)
    a = evaluate_panel(env, agent, [1, 2])
    b = evaluate_panel(env, agent, [1, 2])
    pd.testing.assert_frame_equal(a, b)


def test_summarize_panel_keys_and_values(synthetic_market_data) -> None:
    """Resumen con las claves esperadas y valores coherentes."""
    env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan")
    agent = TWAP(env, batch_size=8, hidden_neurons=4)
    df = evaluate_panel(env, agent, [1, 2, 3])
    summ = summarize_panel(df)
    assert set(summ) == {
        "mean_abs_dev", "max_abs_dev", "off_plan_steps_pct", "completed_pct",
        "time_used_pct", "slippage_bps_mean", "slippage_bps_std", "cum_reward",
    }
    assert summ["mean_abs_dev"] == pytest.approx(df["mean_abs_dev"].mean())
    assert summ["max_abs_dev"] == pytest.approx(df["max_abs_dev"].max())


def test_episode_trace_lengths(synthetic_market_data) -> None:
    """La traza tiene una fila por step y plan/volumen coherentes."""
    env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan")
    agent = TWAP(env, batch_size=8, hidden_neurons=4)
    tr = episode_trace(env, agent, seed=3)
    n = env.state_pos
    assert len(tr) == n
    assert list(tr["step"]) == list(range(1, n + 1))
    assert tr["exec_vol"].iloc[-1] == env.exec_vol
    assert tr["exec_vol"].is_monotonic_increasing
    assert tr["plan"].iloc[-1] == pytest.approx(env._plan_volume(n))
    assert set(tr["action"]) <= {0, 1}


# ---------------------------------------------------------------- exploración


def test_linear_epsilon_decay() -> None:
    """ε lineal baja por episodio, respeta el suelo y la evaluación no la toca."""
    sch = LinearEpsilonDecay(eps_start=1.0, eps_min=0.5, decay=0.2)
    assert sch.on_episode_end() == pytest.approx(0.8)
    assert sch.on_evaluation(0.1) == pytest.approx(0.8)
    assert sch.on_evaluation(100.0) == pytest.approx(0.8)
    for _ in range(10):
        sch.on_episode_end()
    assert sch.epsilon == pytest.approx(0.5)
    state = sch.state_dict()
    other = LinearEpsilonDecay()
    other.load_state_dict(state)
    assert other.epsilon == pytest.approx(0.5)


def test_adaptive_target_epsilon_monotone_and_bounds() -> None:
    """target_epsilon es monótona en el ratio y respeta los extremos."""
    sch = AdaptiveEpsilon(eps_max=1.0, eps_min=0.02, aspiration=1.0, ratio_max=10.0)
    ratios = [0.1, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 10.0, 50.0]
    vals = [sch.target_epsilon(r) for r in ratios]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert sch.target_epsilon(0.5) == pytest.approx(0.02)
    assert sch.target_epsilon(1.0) == pytest.approx(0.02)
    assert sch.target_epsilon(10.0) == pytest.approx(1.0)
    assert sch.target_epsilon(1e6) == pytest.approx(1.0)
    assert 0.02 < sch.target_epsilon(3.0) < 1.0


def test_adaptive_on_episode_end_does_not_change_epsilon() -> None:
    """Jugar episodios sin evaluar no mueve ε."""
    sch = AdaptiveEpsilon()
    sch.on_evaluation(3.0)
    eps = sch.epsilon
    for _ in range(20):
        assert sch.on_episode_end() == eps
    assert sch.epsilon == eps


def test_adaptive_aspiration_never_above_initial() -> None:
    """La aspiración nunca supera la inicial, aunque el ratio sea malo."""
    sch = AdaptiveEpsilon(aspiration=1.0)
    for r in [5.0, 8.0, 3.0, 20.0]:
        sch.on_evaluation(r)
        assert sch.aspiration <= 1.0


def test_adaptive_aspiration_uses_window_median() -> None:
    """Una evaluación afortunada aislada no endurece la aspiración."""
    sch = AdaptiveEpsilon(aspiration=5.0, tolerance=0.1, window=3, smoothing=1.0)
    for _ in range(3):
        sch.on_evaluation(4.0)
    assert sch.aspiration == pytest.approx(4.4)
    sch.on_evaluation(0.1)
    assert sch.aspiration == pytest.approx(4.4)
    sch.on_evaluation(0.1)
    assert sch.aspiration == pytest.approx(0.11)


def test_adaptive_epsilon_can_rise_when_ratio_worsens() -> None:
    """Si el ratio empeora, ε vuelve a subir."""
    sch = AdaptiveEpsilon(eps_max=1.0, eps_min=0.02, aspiration=1.0, smoothing=1.0)
    assert sch.on_evaluation(0.5) == pytest.approx(0.02)
    assert sch.on_evaluation(100.0) == pytest.approx(1.0)
    sch2 = AdaptiveEpsilon(eps_max=1.0, eps_min=0.02, aspiration=1.0, smoothing=0.5)
    low = sch2.on_evaluation(0.5)
    high = sch2.on_evaluation(100.0)
    assert high > low


def test_adaptive_smoothing_moves_gradually() -> None:
    """Con smoothing 0.5, ε recorre la mitad del camino hacia el objetivo."""
    sch = AdaptiveEpsilon(eps_max=1.0, eps_min=0.0, aspiration=1.0, smoothing=0.5)
    assert sch.on_evaluation(0.5) == pytest.approx(0.5)


def test_adaptive_state_dict_roundtrip() -> None:
    """state_dict/load_state_dict restauran ε, aspiración, mejor ratio y ventana."""
    sch = AdaptiveEpsilon(aspiration=5.0)
    for r in [4.0, 3.0, 2.5, 6.0]:
        sch.on_evaluation(r)
    other = AdaptiveEpsilon(aspiration=5.0)
    other.load_state_dict(sch.state_dict())
    assert other.state_dict() == sch.state_dict()
    sch.on_evaluation(2.0)
    other.on_evaluation(2.0)
    assert other.state_dict() == sch.state_dict()


def test_adaptive_best_ratio_zero_gives_no_nan() -> None:
    """Un ratio 0 (seguimiento perfecto) no produce nan."""
    sch = AdaptiveEpsilon()
    eps = sch.on_evaluation(0.0)
    assert np.isfinite(eps)
    assert sch.aspiration == 0.0
    eps = sch.on_evaluation(0.5)
    assert np.isfinite(eps)
    assert np.isfinite(sch.target_epsilon(0.5))


# ---------------------------------------------------------------- entrenamiento


def _train_setup(data, seed: int = 0, adaptive: bool = True):
    """Prepara entorno, agente, scheduler y buffer diminutos."""
    set_seeds(seed)
    env = make_env(data, reward_mode="twap", state_mode="plan", completion_discount=0.9)
    agent = make_agent(env, gamma=0.9)
    sch = AdaptiveEpsilon(eps_max=0.5, eps_min=0.05) if adaptive else LinearEpsilonDecay(0.5, 0.05, 0.1)
    fill_buffer(env, agent, buffer_steps=60, log_every=0)
    return env, agent, sch


def test_train_with_evaluation_outputs_and_checkpoints(synthetic_market_data, tmp_path) -> None:
    """Devuelve history y eval_df con fila inicial y guarda last.pt (y best.pt si mejora)."""
    env, agent, sch = _train_setup(synthetic_market_data)
    history, eval_df = train_with_evaluation(
        env, agent, sch, panel_seeds=[1, 2], baseline_error=1.0, nepisodes=6,
        learn_after=4, update_target_after=50, eval_every=2,
        checkpoint_dir=tmp_path, checkpoint_every=3, verbose=False,
    )
    assert set(history) == {"steps", "rewards", "losses", "epsilons"}
    assert all(len(v) == 6 for v in history.values())
    assert eval_df["episode"].iloc[0] == -1
    assert list(eval_df["episode"]) == [-1.0, 1.0, 3.0, 5.0]
    assert {"ratio", "epsilon", "mean_abs_dev"} <= set(eval_df.columns)
    assert (tmp_path / "last.pt").exists()
    improved = eval_df["ratio"].iloc[1:].min() < eval_df["ratio"].iloc[0]
    assert (tmp_path / "best.pt").exists() == bool(improved)


def test_train_with_evaluation_stops_early(synthetic_market_data, tmp_path) -> None:
    """Con stop_ratio alto, patience 1 y ε en el suelo, para antes de nepisodes y guarda last.pt."""
    set_seeds(0)
    env = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan")
    agent = make_agent(env, gamma=0.9)
    sch = AdaptiveEpsilon(eps_max=0.05, eps_min=0.05)
    fill_buffer(env, agent, buffer_steps=60, log_every=0)
    history, eval_df = train_with_evaluation(
        env, agent, sch, panel_seeds=[1, 2], baseline_error=1.0, nepisodes=20,
        eval_every=1, checkpoint_dir=tmp_path, checkpoint_every=100,
        stop_ratio=1e9, stop_patience=1, verbose=False,
    )
    assert len(history["rewards"]) < 20
    assert len(history["rewards"]) == 1
    assert (tmp_path / "last.pt").exists()


def test_train_with_evaluation_no_stop_without_stop_ratio(synthetic_market_data) -> None:
    """Sin stop_ratio se juegan todos los episodios."""
    env, agent, sch = _train_setup(synthetic_market_data)
    history, _ = train_with_evaluation(
        env, agent, sch, panel_seeds=[1], baseline_error=1.0, nepisodes=3,
        eval_every=1, verbose=False,
    )
    assert len(history["rewards"]) == 3


def test_resume_reproduces_continuous_training(synthetic_market_data, tmp_path) -> None:
    """Parar en last.pt y reanudar equivale al entrenamiento continuo."""
    common = dict(
        panel_seeds=[1, 2], baseline_error=1.0, learn_after=4, update_target_after=50,
        eval_every=2, checkpoint_every=3, verbose=False,
    )
    env, agent, sch = _train_setup(synthetic_market_data, seed=0)
    hist_a, eval_a = train_with_evaluation(env, agent, sch, nepisodes=6, **common)
    weights_a = {k: v.clone() for k, v in agent.primary_model.state_dict().items()}

    env_b, agent_b, sch_b = _train_setup(synthetic_market_data, seed=0)
    train_with_evaluation(env_b, agent_b, sch_b, nepisodes=3, checkpoint_dir=tmp_path, **common)
    assert (tmp_path / "last.pt").exists()

    env_c = make_env(synthetic_market_data, reward_mode="twap", state_mode="plan", completion_discount=0.9)
    agent_c = make_agent(env_c, gamma=0.9)
    sch_c = AdaptiveEpsilon(eps_max=0.5, eps_min=0.05)
    hist_c, eval_c = train_with_evaluation(
        env_c, agent_c, sch_c, nepisodes=6, resume_from=tmp_path / "last.pt", **common
    )
    assert hist_c["steps"] == hist_a["steps"]
    assert hist_c["rewards"] == pytest.approx(hist_a["rewards"])
    assert hist_c["epsilons"] == pytest.approx(hist_a["epsilons"])
    assert np.allclose(hist_c["losses"], hist_a["losses"], equal_nan=True)
    for k, v in agent_c.primary_model.state_dict().items():
        assert np.allclose(v.numpy(), weights_a[k].numpy())
    pd.testing.assert_frame_equal(eval_a, eval_c)


# ---------------------------------------------------------------- experimentos


def test_experiment_config_defaults() -> None:
    """Valores por defecto del Ej. 2 y utilidades de configuración."""
    cfg = ExperimentConfig()
    assert cfg.reward_mode == "twap"
    assert cfg.state_mode == "plan"
    assert cfg.gamma == 0.9
    assert cfg.alpha == 1e-3
    assert cfg.epsilon == "linear"
    assert cfg.target_horizon == 1.0
    assert cfg.stop_ratio is None
    assert len(cfg.panel_seeds) == 20
    assert isinstance(build_scheduler(cfg), LinearEpsilonDecay)
    assert isinstance(build_scheduler(ExperimentConfig(epsilon="adaptive")), AdaptiveEpsilon)
    cfgs = with_seeds(cfg, [1, 2, 3], name="x")
    assert [c.seed for c in cfgs] == [1, 2, 3]
    assert all(c.name == "x" for c in cfgs)
    # la lista por defecto no se comparte entre instancias
    assert ExperimentConfig().panel_seeds is not cfg.panel_seeds


def test_run_many_tiny_config_real_data() -> None:
    """run_many con n_jobs=1 sobre datos reales del repo con una configuración mínima."""
    cfg = ExperimentConfig(
        name="tiny", seed=0, nepisodes=2, buffer_fill=100, batch_size=8,
        buffer_size=500, hidden_neurons=8, panel_seeds=[1, 2], eval_every=1,
    )
    res = run_many([cfg], n_jobs=1)
    assert len(res) == 1
    out = res[0]
    assert set(out) >= {"config", "history", "eval", "baseline_error", "weights"}
    assert len(out["history"]["rewards"]) == 2
    assert list(out["eval"]["episode"]) == [-1.0, 0.0, 1.0]
    assert (out["eval"]["name"] == "tiny").all()
    assert np.isfinite(out["baseline_error"])
