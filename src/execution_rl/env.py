"""Entorno de simulación del libro de órdenes para algoritmos de ejecución."""

import datetime
from typing import Any

import numpy as np
import pandas as pd


class BestExecutionEnv:
    """Simulador del libro de órdenes para la ejecución de una orden CARE de compra."""

    def __init__(
        self,
        data: dict[datetime.date, pd.DataFrame],
        look_back: int = 60,
        min_bins: int = 400,
        max_bins: int = 600,
        min_pct_care: float = 0.075,
        max_pct_care: float = 0.125,
        done_penalty: float = -1.0,
        use_causal_vwap_reward: bool = False,
    ) -> None:
        """Inicializa el entorno con los datos agrupados y parámetros de orden CARE."""
        self.data = data
        self.look_back = look_back
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.min_pct_care = min_pct_care
        self.max_pct_care = max_pct_care
        self.done_penalty = done_penalty
        self.use_causal_vwap_reward = use_causal_vwap_reward

        self.episode_bins: int | None = None
        self.episode_full_len: int | None = None
        self.vol_care: int | None = None

        self.actions_fn = {
            0: self._do_nothing,
            1: self._agg_action,
        }

        self.episode: pd.DataFrame | None = None
        self.episode_full: pd.DataFrame | None = None
        self.episode_vwap: float | None = None
        self.market_ep_vol: pd.Series | None = None
        self.state_pos: int = 0
        self.algo_vwap: float = 0.0
        self.exec_vol: int = 0
        self.actions_hist: list[int] = []
        self.algo_vwap_hist: list[float] = []
        self.market_vwap_hist: list[float] = []
        self.reward_hist: list[float] = []
        self.price_hist: list[float] = []
        self.vol_hist: list[int] = []

        self.n_actions: int = len(self.actions_fn)
        self.n_features: int = self._detect_num_feat()

    def _detect_num_feat(self) -> int:
        """Detecta la dimensión del vector de estado generado por observation_builder."""
        self._reset_env_episode_params()
        self._generate_episode()
        s = self.observation_builder()
        return int(s.shape[0])

    def _reset_env_episode_params(self) -> None:
        """Reinicia las variables internas del episodio antes de comenzar uno nuevo."""
        self.episode_full_len = None
        self.episode = None
        self.episode_full = None
        self.episode_vwap = None
        self.market_ep_vol = None
        self.state_pos = 0
        self.algo_vwap = 0.0
        self.exec_vol = 0
        self.actions_hist = []
        self.algo_vwap_hist = []
        self.market_vwap_hist = []
        self.reward_hist = []
        self.price_hist = []
        self.vol_hist = []

    def _generate_episode_params(self) -> None:
        """(1) Determina aleatoriamente la duración del episodio y el volumen de la orden CARE."""
        self.episode_bins = int(
            np.random.randint(low=self.min_bins, high=self.max_bins + 1)
        )
        pct_bins = float(
            np.random.uniform(low=self.min_pct_care, high=self.max_pct_care)
        )
        self.vol_care = int(pct_bins * self.episode_bins)
        self.episode_full_len = self.episode_bins + self.look_back

        assert self.episode_bins <= self.max_bins
        assert self.episode_bins >= self.min_bins
        assert self.vol_care <= int(self.episode_bins * self.max_pct_care)
        assert self.vol_care >= int(self.episode_bins * self.min_pct_care)
        assert isinstance(self.vol_care, int)

    def _generate_episode(self) -> None:
        """(2) Selecciona aleatoriamente el día y la hora de inicio del episodio."""
        self._generate_episode_params()

        days = list(self.data.keys())
        length_episode = 0
        while length_episode != self.episode_full_len:
            selected_day = np.random.choice(days)
            data_day = self.data[selected_day]
            init_time = np.random.choice(data_day.index)

            hour_pos = data_day.index.get_loc(init_time)
            if isinstance(hour_pos, slice):
                hour_pos = hour_pos.start
            elif isinstance(hour_pos, np.ndarray):
                hour_pos = int(np.flatnonzero(hour_pos)[0])

            initial_position = int(hour_pos) - self.look_back
            final_position = int(hour_pos) + int(self.episode_bins)

            if initial_position < 0 or final_position > len(data_day):
                continue

            self.episode_full = data_day.iloc[initial_position:final_position].copy()
            self.episode = data_day.iloc[int(hour_pos) : final_position].copy()
            length_episode = self.episode_full.shape[0]

    def reset(self) -> np.ndarray:
        """Reinicia el entorno para un nuevo episodio y devuelve la observación inicial."""
        self._reset_env_episode_params()
        self._generate_episode()
        self._compute_episode_market_feat()
        return self.observation_builder()

    def observation_builder(self) -> np.ndarray:
        """(3) Construye el estado con porcentaje de tiempo restante y volumen restante."""
        time_left = (self.episode_bins - self.state_pos) / self.episode_bins
        vol_left = 1.0 - (self.exec_vol / self.vol_care)
        return np.array([time_left, vol_left], dtype=np.float32)

    def _compute_episode_market_feat(self) -> tuple[float, pd.Series]:
        """(4) Calcula el volumen por bin y el VWAP de mercado del episodio completo."""
        mid = (self.episode["ask1"] + self.episode["bid1"]) / 2.0
        # clip: cumvol tiene saltos negativos puntuales en los datos
        self.market_ep_vol = self.episode["cumvol"].diff().clip(lower=0)
        self.market_ep_vol.iloc[0] = 0.0
        cum_vol = float(self.market_ep_vol.sum())
        if cum_vol == 0.0:
            self.episode_vwap = float(mid.mean())
        else:
            self.episode_vwap = float(
                (mid.iloc[:-1].values * self.market_ep_vol.iloc[1:].values).sum()
                / cum_vol
            )
        return self.episode_vwap, self.market_ep_vol

    def _compute_market_vwap(self) -> float:
        """(5) Calcula el VWAP de mercado acumulado hasta el step actual sin fuga de futuro."""
        mid_p = (self.episode["ask1"] + self.episode["bid1"]) / 2.0
        v = self.market_ep_vol
        # vol[i] se valora a mid[i-1] (mismo criterio que episode_vwap)
        p_arr = mid_p.values[: self.state_pos]
        v_arr = v.values[1 : self.state_pos + 1]
        sum_vol = float(np.sum(v_arr))
        if sum_vol == 0.0:
            return float(mid_p.values[self.state_pos])
        return float(np.sum(p_arr * v_arr) / sum_vol)

    def _compute_reward(self, price: float, vol: float) -> float:
        """(6) Calcula la recompensa inmediata tras ejecutar una acción en el mercado."""
        if vol == 0:
            return 0.0
        ref_vwap = (
            self._compute_market_vwap()
            if self.use_causal_vwap_reward
            else self.episode_vwap
        )
        return float((ref_vwap - price) / ref_vwap)

    def _agg_action(self) -> float:
        """(7) Ejecuta una compra agresiva de 1 título al mejor precio de venta (ask1)."""
        price = float(self.episode["ask1"].values[self.state_pos])
        vol = 1
        self.price_hist.append(price)
        self.exec_vol += vol
        self.vol_hist.append(vol)
        self.actions_hist.append(1)

        algo_vwap = self._compute_algo_vwap()
        self.algo_vwap = algo_vwap
        self.algo_vwap_hist.append(algo_vwap)
        return self._compute_reward(price, vol)

    def _compute_algo_vwap(self) -> float:
        """(8) Calcula el VWAP acumulado por las ejecuciones del algoritmo hasta el step actual."""
        p_arr = np.asarray(self.price_hist, dtype=np.float64)
        v_arr = np.asarray(self.vol_hist, dtype=np.float64)
        total_vol = float(np.sum(v_arr))
        if total_vol == 0.0:
            return 0.0
        return float(np.sum(p_arr * v_arr) / total_vol)

    def _do_nothing(self) -> float:
        """(9) No ejecuta orden en el step actual y mantiene el VWAP previo del algoritmo."""
        price = 0.0
        vol = 0
        self.price_hist.append(price)
        self.vol_hist.append(vol)
        self.actions_hist.append(0)

        algo_vwap = self.algo_vwap_hist[-1] if self.algo_vwap_hist else 0.0
        self.algo_vwap = algo_vwap
        self.algo_vwap_hist.append(algo_vwap)
        return self._compute_reward(price, vol)

    def _compute_stop_conditions(self) -> tuple[bool, bool]:
        """(10) Comprueba si se ha agotado el tiempo o si se ha completado la orden CARE."""
        is_bins_complete = self.state_pos == self.episode_bins
        is_ord_complete = self.exec_vol == self.vol_care
        return is_bins_complete, is_ord_complete

    def _compute_done(self) -> bool:
        """(11) Determina si el episodio ha finalizado por tiempo o por volumen completado."""
        is_bins_complete, is_ord_complete = self._compute_stop_conditions()
        return bool(is_bins_complete or is_ord_complete)

    def _compute_done_reward(self) -> float:
        """(12) Asigna recompensa adicional al cierre del episodio según si se completó la orden."""
        _, is_ord_complete = self._compute_stop_conditions()
        return 0.0 if is_ord_complete else float(self.done_penalty)

    def action_sample(self) -> int:
        """(13) Muestra una acción aleatoria ponderada por la tasa media de ejecución."""
        p = float(self.vol_care / self.episode_bins)
        return int(np.random.choice([0, 1], p=[1.0 - p, p]))

    def step(self, action: int) -> tuple[np.ndarray | None, float, bool, dict[str, Any]]:
        """Aplica la acción en el entorno y devuelve observación, recompensa, done e info."""
        market_vwap = self._compute_market_vwap()

        act_fn = self.actions_fn.get(int(action))
        if act_fn is None:
            raise ValueError(
                f"Invalid action {action}. Valid actions {list(self.actions_fn.keys())}"
            )

        reward = act_fn()

        self.market_vwap_hist.append(market_vwap)
        self.reward_hist.append(reward)

        self.state_pos += 1
        done = self._compute_done()

        if done:
            reward += self._compute_done_reward()
            self.reward_hist[-1] = reward
            return None, float(reward), done, {}

        observation = self.observation_builder()
        return np.array(observation, dtype=np.float32), float(reward), done, {}

    def stats_df(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Construye los DataFrames de evolución del algoritmo y del mercado para graficar."""
        my_df = pd.DataFrame(
            {"vwap": self.algo_vwap_hist, "vol": self.vol_hist},
            index=list(self.episode.index)[: len(self.algo_vwap_hist)],
        )
        my_df = my_df.reindex(self.episode.index)
        my_df["vol"] = my_df["vol"].fillna(0)
        my_df["vwap"] = my_df["vwap"].ffill()

        mpx = (self.episode["ask1"] + self.episode["bid1"]) / 2.0
        # vol[i] valorado a mid[i-1]; termina en episode_vwap
        p = mpx.shift(1)
        v = self.market_ep_vol
        denom = np.where(v.cumsum() == 0, np.nan, v.cumsum())
        market_vwap = ((p * v).fillna(0).cumsum() / denom).fillna(mpx)
        market_df = pd.DataFrame(
            {"vwap": market_vwap, "vol": v},
            index=v.index,
        )

        return my_df, market_df, mpx
