"""Esquemas de exploración ε: decaimiento por episodios (referencia) y ε guiada por rendimiento (Ej. 3)."""

from typing import Any

import numpy as np


class LinearEpsilonDecay:
    """ε lineal por episodio, como el taller: depende solo del nº de episodios jugados."""

    def __init__(self, eps_start: float = 1.0, eps_min: float = 0.05, decay: float = 0.005) -> None:
        """Fija ε inicial, suelo y cuánto baja por episodio."""
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.decay = decay

    def on_episode_end(self) -> float:
        """Baja ε un paso fijo al acabar cada episodio."""
        self.epsilon = max(self.epsilon - self.decay, self.eps_min)
        return self.epsilon

    def on_evaluation(self, ratio: float) -> float:
        """El rendimiento no influye en ε."""
        return self.epsilon

    def state_dict(self) -> dict[str, float]:
        """Estado serializable para los checkpoints."""
        return {"epsilon": self.epsilon}

    def load_state_dict(self, state: dict[str, float]) -> None:
        """Restaura el estado desde un checkpoint."""
        self.epsilon = float(state["epsilon"])


class AdaptiveEpsilon:
    """ε guiada por el rendimiento frente al benchmark TWAP y un nivel de aspiración que se endurece.

    ratio = error de seguimiento del agente / error del TWAP determinista sobre el mismo panel.
    ε objetivo interpola en escala logarítmica entre eps_max (ratio >= ratio_max) y eps_min (ratio <= aspiración).
    """

    def __init__(
        self,
        eps_max: float = 1.0,
        eps_min: float = 0.02,
        aspiration: float = 1.0,
        ratio_max: float = 10.0,
        smoothing: float = 0.5,
        tolerance: float = 0.1,
        window: int = 3,
    ) -> None:
        """Fija los extremos de ε, la aspiración inicial, el suavizado, la holgura y la ventana de la mediana."""
        self.epsilon = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.aspiration0 = aspiration
        self.aspiration = aspiration
        self.ratio_max = ratio_max
        self.smoothing = smoothing
        self.tolerance = tolerance
        self.window = window
        self.best_ratio = np.inf
        self.recent: list[float] = []

    def target_epsilon(self, ratio: float) -> float:
        """Traduce el ratio de error a ε: lejos del benchmark se explora, al alcanzar la aspiración no."""
        aspiration = max(self.aspiration, 1e-6)
        if ratio <= aspiration:
            return self.eps_min
        span = np.log(self.ratio_max / aspiration)
        frac = float(np.clip(np.log(ratio / aspiration) / span, 0.0, 1.0))
        return self.eps_min + (self.eps_max - self.eps_min) * frac

    def on_episode_end(self) -> float:
        """Sin evaluación nueva no hay información: ε no cambia por jugar más episodios."""
        return self.epsilon

    def on_evaluation(self, ratio: float) -> float:
        """Actualiza la aspiración (trinquete sobre la mediana reciente del ratio) y mueve ε hacia su objetivo."""
        # mediana de las últimas evaluaciones: una evaluación afortunada no fija el listón
        self.recent = (self.recent + [ratio])[-self.window :]
        self.best_ratio = min(self.best_ratio, float(np.median(self.recent)))
        # aspiración: nunca peor que la inicial; se endurece hasta el mejor ratio con holgura
        self.aspiration = min(self.aspiration0, self.best_ratio * (1.0 + self.tolerance))
        target = self.target_epsilon(ratio)
        self.epsilon = (1.0 - self.smoothing) * self.epsilon + self.smoothing * target
        return self.epsilon

    def state_dict(self) -> dict[str, Any]:
        """Estado serializable para los checkpoints."""
        return {
            "epsilon": self.epsilon,
            "aspiration": self.aspiration,
            "best_ratio": self.best_ratio,
            "recent": list(self.recent),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restaura el estado desde un checkpoint."""
        self.epsilon = float(state["epsilon"])
        self.aspiration = float(state["aspiration"])
        self.best_ratio = float(state["best_ratio"])
        self.recent = [float(r) for r in state.get("recent", [])]
