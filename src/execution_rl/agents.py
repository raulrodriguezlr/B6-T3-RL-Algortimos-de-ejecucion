"""Agentes de ejecución: Double Deep Q-Network (PyTorch) y referencia TWAP."""

from collections import deque
from typing import Any

import numpy as np
import torch
from torch import nn

from .env import BestExecutionEnv


class QNetwork(nn.Module):
    """Red neuronal multicapa que aproxima la función de valor acción Q(s, a)."""

    def __init__(self, n_features: int, n_actions: int, hidden_neurons: int = 32) -> None:
        """Define dos capas ocultas densas con activación ReLU y capa de salida lineal."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, int(hidden_neurons)),
            nn.ReLU(),
            nn.Linear(int(hidden_neurons), int(hidden_neurons)),
            nn.ReLU(),
            nn.Linear(int(hidden_neurons), n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula los valores Q para cada acción dado el tensor de estado."""
        return self.net(x)


class DDQNAgent:
    """Agente Double DQN con buffer de repetición de experiencias combinado (CER)."""

    def __init__(
        self,
        env: BestExecutionEnv,
        gamma: float = 0.99,
        epsilon: float = 0.9,
        alpha: float = 0.01,
        batch_size: int = 64,
        buffer_size: int = 50000,
        hidden_neurons: int = 32,
        trainable: bool = True,
        force_first_buy: bool = True,
    ) -> None:
        """Inicializa las redes primaria y objetivo, el optimizador Adam y el buffer."""
        self.env = env
        self.na = env.n_actions
        self.n_features = env.n_features
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.minus_factor = None
        self.div_factor = None

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray | None]] = deque(
            maxlen=self.buffer_size
        )
        self.hidden_neurons = hidden_neurons
        self.force_first_buy = force_first_buy

        self.primary_model = self.network()
        self.target_model = self.network()
        self.update_target()
        self.target_model.eval()

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.primary_model.parameters(), lr=alpha)
        self._trainable = trainable

    def network(self) -> QNetwork:
        """Construye una instancia de la arquitectura QNetwork configurada para el entorno."""
        return QNetwork(
            n_features=self.n_features,
            n_actions=self.na,
            hidden_neurons=self.hidden_neurons,
        )

    def experience(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s1: np.ndarray | None,
        done: bool = False,
    ) -> None:
        """Almacena una transición (s, a, r, s1) en la memoria de repetición."""
        if done:
            s1 = None
        self.buffer.append((s, int(a), float(r), s1))

    def sample(self) -> list[tuple[np.ndarray, int, float, np.ndarray | None]]:
        """Muestrea un lote mediante Combined Experience Replay incluyendo la última transición."""
        if len(self.buffer) < self.batch_size:
            raise ValueError(
                f"Buffer insuficiente ({len(self.buffer)}) para batch_size={self.batch_size}."
            )
        idx = np.hstack(
            [
                np.random.choice(
                    len(self.buffer) - 1,
                    size=self.batch_size - 1,
                    replace=False,
                ),
                len(self.buffer) - 1,
            ]
        )
        return [self.buffer[int(i)] for i in idx]

    @staticmethod
    def getQs(s: np.ndarray, model: nn.Module) -> np.ndarray:
        """Evalúa el modelo sin gradientes y devuelve los valores Q como array de NumPy."""
        if len(s) == 0:
            return np.empty((0, 0), dtype=np.float32)
        s_arr = np.atleast_2d(np.asarray(s, dtype=np.float32))
        with torch.no_grad():
            tensor_s = torch.from_numpy(s_arr)
            return model(tensor_s).cpu().numpy()

    def act(self, s: np.ndarray) -> int:
        """Selecciona una acción aplicando política epsilon-greedy o forzado inicial."""
        if self.force_first_buy and self.env.state_pos == 0:
            return 1
        if self._trainable:
            if np.random.rand() > self.epsilon:
                a = self._act_no_explore(s)
            else:
                a = self.env.action_sample()
        else:
            a = self._act_no_explore(s)
        return int(a)

    def set_trainable(self, train: bool = False) -> None:
        """Activa o desactiva el modo de exploración durante la selección de acciones."""
        self._trainable = train

    def _act_no_explore(self, s: np.ndarray) -> int:
        """Elige la acción con mayor valor Q estimado por la red principal."""
        q = self.getQs(
            s=np.asarray(s, dtype=np.float32).reshape(1, -1),
            model=self.primary_model,
        )
        return int(np.argmax(q))

    def getQ(self, s: np.ndarray, a: int) -> np.ndarray:
        """Devuelve el valor Q estimado por la red principal para el estado y acción dados."""
        return self.getQs(np.atleast_2d(s), self.primary_model)[:, a]

    def update_target(self) -> None:
        """Copia los pesos de la red principal en la red objetivo."""
        self.target_model.load_state_dict(self.primary_model.state_dict())

    def learn(self) -> float:
        """Actualiza la red principal usando la regla de actualización de Double DQN."""
        batch = self.sample()
        s_list = np.array([exp[0].tolist() for exp in batch], dtype=np.float32)
        s1_list = np.array(
            [exp[3].tolist() for exp in batch if exp[3] is not None],
            dtype=np.float32,
        )

        qs = self.getQs(s_list, self.primary_model).copy()
        if len(s1_list) > 0:
            # La red principal selecciona la mejor acción siguiente y la objetivo la evalúa
            qs1 = self.getQs(s1_list, self.primary_model)
            q_target = self.getQs(s1_list, self.target_model)
        else:
            qs1 = np.empty((0, self.na), dtype=np.float32)
            q_target = np.empty((0, self.na), dtype=np.float32)

        targets = qs
        k = 0
        for i, (_, a, r, s1) in enumerate(batch):
            if s1 is None:
                targets[i, a] = r
            else:
                best_next_action = int(np.argmax(qs1[k]))
                targets[i, a] = r + self.gamma * q_target[k, best_next_action]
                k += 1

        s_tensor = torch.from_numpy(s_list)
        target_tensor = torch.from_numpy(targets)

        self.primary_model.train()
        self.optimizer.zero_grad()
        preds = self.primary_model(s_tensor)
        loss = self.loss_fn(preds, target_tensor)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())


class TWAP(DDQNAgent):
    """Agente determinista TWAP que mantiene el volumen restante alineado con el tiempo restante."""

    def __init__(self, env: BestExecutionEnv, horizon: float = 1.0, **kwargs: Any) -> None:
        """horizon < 1 comprime el calendario para terminar en esa fracción del tiempo (Ej. 4B)."""
        super().__init__(env, **kwargs)
        self.horizon = horizon

    def act(self, s: np.ndarray) -> int:
        """Compra (1) si el volumen pendiente iguala o supera al pendiente del plan; si no, espera."""
        # tiempo restante del plan: 0 a partir de horizon
        plan_time_left = max(0.0, (float(s[0]) - (1.0 - self.horizon)) / self.horizon)
        return 1 if float(s[1]) >= plan_time_left else 0
