# RL aplicado a algoritmos de ejecución (MIAX B6-T3)

Agente Double DQN (PyTorch) que aprende a ejecutar una orden CARE de compra, troceada en órdenes hijas de 1 título, sobre un simulador del libro de órdenes (REP, primera posición, bins de 5 s, 13:00 a 17:25). El agente replica un TWAP solo a través de la recompensa y se amplía con exploración adaptativa, horizonte del 90 %, parada por aprendizaje y estado de mercado.

## Estructura

| Ruta | Contenido |
|---|---|
| `notebooks/ej1_entorno_y_entrenamiento.ipynb` | Ej. 1: entorno `BestExecutionEnv` completo, DDQN en PyTorch, bucle `reset, act, step, experience, learn, update_target` |
| `notebooks/ej2_twap_recompensas.ipynb` | Ej. 2: estado, recompensa `-κ·|desvío frente al plan|`, parametría motivada y barrido |
| `notebooks/ej3_exploracion_adaptativa.ipynb` | Ej. 3: ε ligada al ratio frente al TWAP determinista |
| `notebooks/ej4_ampliaciones.ipynb` | Ej. 4: ideas A (estado de mercado), B (horizonte 90 %) y C (parada, checkpoints y reanudación) |
| `src/execution_rl/` | Código reutilizable: `env.py`, `agents.py`, `train.py`, `exploration.py`, `evaluation.py`, `experiments.py`, `report.py`, `data.py` |
| `data/` | Datos del taller convertidos a `csv.gz` (`{rep,san}_{train,val,test}`), legibles con pandas actual |
| `results/` | Figuras y tablas generadas por los notebooks |
| `tests/` | Tests con pytest, sin red |
| `presentacion/` | Presentación de la exposición (PDF) |

Los bloques modificados respecto al notebook del taller se señalan en cada notebook con `[BLOQUE MODIFICADO]` o en una tabla al inicio.

## Resultados principales

| Ejercicio | Resultado |
|---|---|
| 2 | Desvío medio frente al plan de 0,25 títulos (TWAP determinista: 0,41), 100 % de órdenes completas en días, horas, duraciones (400 a 600 bins) y tamaños (7,5 a 12,5 %) distintos |
| 3 | ε baja de 0,1 entre los episodios 21 y 26 (lineal: 135) con la misma calidad final; recompensa media de entrenamiento -40 frente a -81 (5 semillas) |
| 4B | Ejecución terminada en el 89,2 % del tiempo con 100 % de órdenes completas |
| 4C | Parada en el episodio 54 de 400; la reanudación desde `last.pt` reproduce el entrenamiento continuo |
| 4A | Mejora de 0,27 a 0,36 pbs de slippage frente al TWAP en train, a costa de más desvío y órdenes con 1 título pendiente |

## Ejecución

Requisitos: Python 3.12 o superior con `numpy`, `pandas`, `matplotlib`, `torch` y `pytest`. Las versiones fijadas en el material del taller (TensorFlow 2.5) no instalan en un Python actual; por eso el agente está en PyTorch y los `.pkl` originales se convirtieron a `csv.gz`.

Ejecutar un notebook desde `notebooks/`:

```bash
python -m nbconvert --to notebook --execute --inplace ej1_entorno_y_entrenamiento.ipynb
```

Tests desde la raíz:

```bash
python -m pytest -q
```

Las semillas están fijadas (`set_seeds`, `reset(seed)`), de modo que los resultados se reproducen.

## Referencias

Material del taller `best-execution` (Oteo García y Sanz del Real). Sutton y Barto, *Reinforcement Learning: An Introduction*, 2.ª ed.
