# CLAUDE.md: práctica B6-T3 RL aplicado a algoritmos de ejecución

Contexto y reglas de trabajo para esta práctica. Manda sobre el CLAUDE.md global solo donde lo contradiga de forma explícita; en todo lo demás valen las reglas globales (resumidas en la sección 8).

## 1. Contexto

- Asignatura: Aprendizaje por Refuerzo aplicado a finanzas, MIAX, Bloque 6, T3 (taller de algoritmos de ejecución). Profesor: Jesús Sanz del Real, CFA (Oliver Wyman), jesus.sanz@oliverwyman.com / jesussanzdr@gmail.com.
- Material de clase (solo lectura) en `docs/`:
  - `enunciado_practica.md`: enunciado de los 4 ejercicios.
  - `ejercicios_taller_rl.docx`: enunciado original.
  - `guia_estudio.md`: guía de repaso (teoría, descripción del entorno, ideas por ejercicio).
  - `casos_practicos.pdf`: caso de ejecución (TWAP/VWAP/POV, estado, acciones, recompensas).
  - `applying_RL_MIAX.pdf`: cómo ajustar ε, α, γ, buffer y recompensas.
  - `rl-material.zip`: notebooks de algoritmos (TD, deep, double, dueling, rainbow...). Los dos importantes según el profesor: `TD-q-learning` y `deep`.
  - `best-execution.zip`: material del taller (ver sección 4). Contiene un `.git` propio: no se descomprime dentro del repo tal cual.
  - `transcripciones.txt.txt`: VTT de las clases (~690 KB). La explicación de la práctica está hacia el final (hacia la línea 18800 en adelante).
  - `Orden Taller v2.pdf`: orden de exposición del 18/11.
- Referencia: Sutton & Barto, 2.ª ed. Código de partida: Oteo García y Sanz del Real (`best-execution`).

## 2. Qué pide la práctica

Agente de RL que aprende a ejecutar una **orden CARE de compra** troceada en órdenes hijas de tamaño 1 sobre un simulador del libro de órdenes (primera posición, bins de 5 s, sesiones 13:00 a 17:25), y que acaba replicando un **TWAP**. Métrica del sector: VWAP slippage en pbs, con `VWAP = Σ pᵢ·vᵢ / Σ vᵢ`.

| Ej. | Puntos | Contenido |
|---|---|---|
| 1 | 3,5 | Entorno y entrenamiento completos de extremo a extremo, orden correcto de interacción, decisiones documentadas. Se puede rehacer entorno y notebook, o cambiar TF por PyTorch, usando los archivos del taller. |
| 2 | 3,5 | Recompensas para replicar TWAP (explicar estado, recompensa y por qué, con gráficos) y parametría (γ, buffer, etc., motivada). Hijas de tamaño 1, sin impacto ni costes. Debe funcionar con distintos días, momentos, duraciones y tamaños de orden. Basta con train; comportamiento similar a `images/twap_model/twap_reproduction_dqn.png`. |
| 3 | 1,5 | ε independiente del nº de episodios, que baje según mejora el agente. Responder: ¿cómo se determina que ha mejorado? ¿cómo se traslada a ε? Puede mantenerse un nº fijo de episodios. |
| 4 | 1,5 (opcional) | Ampliación: A) mejorar VWAP/dispersión con estado ampliado (libro, tendencia); B) penalizar volumen pendiente y consumir ~90 % del tiempo; C) parada por aprendizaje, guardado periódico de pesos y recuperación. Libertad para otras mejoras vistas en clase. |

Entrega:
- **Notebook ejecutado** con la resolución y la motivación de cada punto. Memoria aparte opcional y breve.
- **Exposición el miércoles 18/11/2026**, 15 minutos por grupo, hueco desde las 18:30 (ver `Orden Taller v2.pdf`; el grupo se identifica por el nombre de un compañero).
- Evaluación: pesan sobre todo **exposición y motivación**. El código respalda lo expuesto y puede ajustar la nota.
- Se señalan los bloques modificados en cada ejercicio y se explica la lógica. No hay solución única.

## 3. Lo que dijo el profesor (transcripción, relevante para la práctica)

- Ej. 1: basta con una ejecución completa hasta el final del notebook. No hace falta que la recompensa dependa del rendimiento.
- Ej. 2: el estado `[tiempo restante, volumen restante]` basta para un TWAP, sin features de mercado. La regla determinista es: si se va por delante del plan, no se ejecuta; si se está por debajo o igual, se lanza otra orden hija. Se pide además lograrlo **por recompensas**. Con 2 o 3 escenarios, incluso un Q-learning tabular lo aprende.
- El entorno no es obligatorio: se puede rehacer entero. Si se añaden funciones de impacto u otras mejoras, se valora al alza.
- Ej. 3: tocar el decaimiento de ε para que no sea simplemente proporcional al nº de episodios.
- Ej. 4 (sugerencia concreta): trabajar las recompensas para que no se ejecute hasta el final sino en un rango del 70 al 90 % del tiempo.
- Parámetros (ya en `applying_RL_MIAX.pdf` y clase): γ depende del nº de steps (`0.99^500 ≈ 0.7 %`); α nunca decae a 0 (problema no estacionario); buffer: independencia frente a ruido (las primeras experiencias son casi aleatorias); rellenar el buffer antes de entrenar.
- Critica el decaimiento de ε por nº de episodios (arbitrario). Prefiere condicionarlo a un benchmark (TWAP) y a un nivel de aspiración.
- Recompensas: lo más importante es aterrizar el objetivo; magnitud y signo importan; comprobar el comportamiento, no solo la loss.
- Dudas de alcance: escribirle antes de atascarse en un concepto tonto.

## 4. El material del taller (verificado leyendo el código)

Estructura de `best-execution/` (zip): `data/` (`rep_data.pkl`, `san_data.pkl`, `new_version/` con `orderbook.pkl` y `executions.pkl`), `src/agents/dqn.py`, `src/report/report.py`, `workshop/execution_lab.ipynb` (con `"-----"` y `TODO`), `workshop/execution_lab-step_by_step.ipynb`, `images/twap_model/`.

**Datos.** `dict_` con claves `train` (60 primeros días hábiles de 2018), `validation` (30 primeros de 2019) y `test` (días 31 a 60 de 2019). Cada valor es un DataFrame por día con `bid1`, `ask1`, `cumvol`. Con REP o SAN basta. Los ficheros de `new_version/` usan otros intervalos (train 40 días, val 20, test 20 de 2019).

**`BestExecutionEnv(data, look_back=60)`**, con el orden `(1)` a `(13)` de los docstrings:
1. `_generate_episode_params`: `episode_bins` entero en [400, 600]; `vol_care = int(U[0.075, 0.125] · episode_bins)`.
2. `_generate_episode`: día aleatorio, hora de inicio aleatoria, `episode_full` (con `look_back`) y `episode`; repite hasta que la longitud sea la correcta.
3. `observation_builder`: `[tiempo restante %, volumen restante %]`.
4. `_compute_episode_market_feat`: mid-price, volumen de mercado por bin, `episode_vwap`.
5. `_compute_market_vwap`: VWAP de mercado hasta el step.
6. `_compute_reward`: por defecto `(episode_vwap - price) / episode_vwap`, con caso especial si `vol == 0`.
7. `_agg_action` (compra 1 título a `ask1`), 9. `_do_nothing`, 8. `_compute_algo_vwap`.
10 a 12. `_compute_stop_conditions`, `_compute_done` (tiempo agotado o orden completa), `_compute_done_reward` (libre).
13. `action_sample` (acción aleatoria con probabilidad `p`, no tiene por qué ser uniforme).
- `step()` devuelve `(obs, reward, done, {})`; al terminar la observación es `None`. Acciones: 0 no hacer nada, 1 compra agresiva.
- `stats_df()` y `plot_results(env)` generan el gráfico mercado frente a algoritmo. `class TWAP(DDQNAgent)` con `act(s)` determinista es un TODO del notebook.

**`DDQNAgent`** (Keras/TF): 2 capas densas relu + salida lineal, red primary y target (Double DQN), buffer `deque`, muestreo CER (el último elemento siempre entra en el batch), `train_on_batch`. Por defecto γ=0.99, ε=0.9, α=0.01, batch=64, buffer=50 000, 32 neuronas.

**Bucle del notebook** (el "orden correcto" del Ej. 1): crear env y agente; llenar el buffer con acciones aleatorias (la celda dice "eps es 1", pero el agente arranca con ε=0.9: hay que fijarlo); `set_trainable(True)`; por episodio: `reset`, `act`, `step`, `experience`, cada `learn_after` steps `learn`, cada `update_target_after` steps `update_target`; al terminar el episodio, decaer ε y guardar métricas; evaluar con `set_trainable(False)` y `plot_results`.

Puntos a vigilar en el código base:
- `DDQNAgent.act` devuelve siempre 1 cuando `state_pos == 0` (primer step forzado a comprar). Decidir si se conserva y justificarlo.
- `getQ` llama a `getQs(s)` sin el parámetro `model`: está roto, pero no se usa.
- `mse` se usa en el bucle antes de definirse si todavía no ha habido `learn`.
- `sample()` usa `replace=False` con `len(buffer) - 1`: exige buffer mayor que `batch_size`.
- El episodio contiene `episode_bins` steps; la orden tiene entre 7,5 % y 12,5 % de ese valor en títulos: un TWAP ejecuta 1 de cada `episode_bins / vol_care` steps (entre 8 y 13).
- `episode_vwap` y `stats_df` pueden usar un precio distinto (mid frente a `ask1`): conviene unificar al comparar.
- Dependencias del zip: `tensorflow==2.5.0`, `numpy==1.19.2`, `pandas==1.4.0`, Python 3.9 a 3.11. No instalan en Python 3.14.

## 5. Cómo programa el profesor (para que el código encaje)

- Notebooks, clases con docstring en español, nombres en inglés, `numpy`/`pandas`/`matplotlib`.
- Bucle explícito `reset, act, step, experience, learn, update_target`, métricas por episodio en listas (`history_rewards`, `history_losses`...), log cada `n_log` episodios.
- Gráficos: `plot_results(env)` (precio y volumen acumulado frente a mercado) y media móvil de la recompensa por episodio.

Para este repo se mantiene ese estilo, cumpliendo las reglas de la sección 8.

## 6. Estructura propuesta del repo (a confirmar con el usuario antes de crearla)

```
src/execution_rl/
  env.py           # BestExecutionEnv completado, con las recompensas configurables
  agents.py        # DDQN (PyTorch o Keras), TWAP determinista de referencia
  train.py         # relleno de buffer, bucle, ε adaptativo, checkpoints
  report.py        # plot_results y gráficos de resultados
notebooks/         # ej1, ej2, ej3, ej4 (o uno solo con secciones señaladas)
tests/             # pytest, sin red
results/           # figuras y tablas generadas
data/              # pkl descomprimidos del zip (no se versionan)
```

Antes de crear nada hay que decidir con el usuario: notebook único frente a uno por ejercicio, y TF frente a PyTorch.

## 7. Criterios de experimentación

- Semillas fijas y separadas (`np.random.seed` y semilla del framework). Ver comportamiento por semilla, no solo una.
- Ajuste de hiperparámetros con `train` (el enunciado no exige validación ni test en el Ej. 2). Si se evalúa fuera de train, `validation` para ajustar y `test` solo para informar.
- Cada gráfico del Ej. 2 se comprueba con varios días, horas de inicio, duraciones y tamaños de orden, no con un episodio afortunado.
- Medidas contra el TWAP determinista: error entre volumen acumulado y la diagonal, nº de órdenes fuera de plan, % de la orden completada, VWAP slippage en pbs (media y dispersión).
- Mirar la **recompensa media móvil**, no la loss.
- Todo hiperparámetro va como parámetro con valor por defecto explícito. Cada conclusión de la exposición se apoya en un número o figura generado por el código.

## 8. Reglas de trabajo (del CLAUDE.md global, aplicadas aquí)

Cadena obligatoria cada vez que se toca código, salvo que el usuario diga "sin cadena":

| Paso | Quién | Qué hace |
|---|---|---|
| 0. Plan | Agente principal (Opus) con el usuario | Lee el código, acota la tarea, escribe el plan. Decide y orquesta. |
| 1. Código | `dev-coder` (Sonnet) | Implementa el plan y nada más. |
| 2. Auditoría | `dev-auditor` (Sonnet) | Regresiones, corrección, fuga de datos (train/test), alcance, malas prácticas. No edita. |
| 3. Corrección | `dev-fixer` (Sonnet) | Aplica solo los hallazgos del auditor. |
| 4. QA | `dev-qa` (Sonnet) | Ejecuta de punta a punta y prueba casos reales y bordes. No edita. |
| 5. Test | `dev-tester` (Sonnet) | Escribe y ejecuta tests; reporta la salida tal cual. |

- Subagentes en serie, siempre Sonnet, nunca en paralelo sobre los mismos ficheros.
- BLOQUEANTE del auditor: tras el fixer se repite la auditoría una vez.
- Fallos de QA o tester vuelven al fixer; tras dos vueltas sin arreglo se para y se avisa.
- Al final el agente principal revisa el diff y resume.
- Briefs e informes entre agentes en modo caveman; código, rutas, comandos y errores exactos.
- En esta práctica el auditor revisa en especial: que el entorno respete el orden de interacción (`reset`, `act`, `step`, `experience`, `learn`), que no haya fuga de información de futuro en estado o recompensa, semillas, coherencia entre `vol_care` y nº de steps, y que los bloques modificados del notebook base estén señalados.

Commits y PR:
- Prohibido atribuir nada a Claude o Anthropic (ni `Co-Authored-By`, ni pie "Generated with Claude Code").
- Mensaje en imperativo, explicable en una línea. Si no cabe, son dos commits.
- No se hace commit ni push sin que el usuario lo pida.

Comentarios en el código:
- Breves, estilo telegrama, solo para orientar o aclarar lo no obvio. No se comenta lo que el código ya dice.
- Docstrings de una o dos frases, sin secciones `Parameters` / `Returns`; los tipos van en la firma.

Redacción (comentarios, markdown, celdas de notebook, memoria):
- Prohibida la primera persona: se usa la forma impersonal con "se" ("se entrena", "se observa").
- Ninguna alusión a la IA.
- Sin relleno, sin listas de emojis, sin tono de folleto.
- Prohibidos el guion largo y el semilargo; se usan paréntesis, comas o dos puntos.
- Las celdas markdown del notebook explican la motivación (qué se hace y por qué), porque es lo que se evalúa.

Buenas prácticas:
- Type hints en funciones públicas.
- Sin rutas absolutas en el código: todo relativo a la raíz del repo o desde config (el notebook base usa `../data/...`).
- Umbrales como parámetros con valor por defecto explícito.
- Semillas fijas en todo lo aleatorio.
- Los tests no tocan la red.
- Refactorizar cuando haga falta, sin inflar cambios pequeños.
- No se instala una dependencia sin decirlo antes.

## 9. Entorno local

- Python 3.14 por defecto (hay también 3.13, 3.12 y 3.10). Instalados: numpy 2.4.4, pandas 3.0.2, matplotlib 3.11.0, pytest 9.0.3, **torch 2.9.0**. **TensorFlow y Keras no están instalados** y las versiones fijadas por el zip no corren en 3.14.
- Opciones para el agente: portar `DDQNAgent` a PyTorch (permitido por el enunciado, es lo que encaja con el entorno actual) o crear un entorno Python 3.11 con TF. Decisión pendiente con el usuario.
- **Datos (verificado 17/11/2026):** los `.pkl` originales (pandas 1.4) **no cargan** con pandas 3.0 ni 2.3 (`TypeError: ... 'placement' ... got slice`). Cargan con `pandas==1.4.0` + `numpy==1.22.4` en Python 3.10. Se convirtieron a `data/{rep,san}_{train,val,test}.csv.gz` (índice `time`, columnas `ask1, bid1, vask1, vbid1, last, cumvol, date`, sin NaN, 60/30/30 días, 3177 a 3181 bins por día). Se leen con `pd.read_csv(..., index_col="time", parse_dates=["time"])` y `groupby("date")` devuelve el dict de días que espera el entorno.
- En el pickle original la clave de validación es `val`, no `validation`, y las claves de día son `datetime.date`.
- Venvs temporales usados para la conversión (py3.12 y py3.10) en el scratchpad de la sesión; no forman parte del repo. El notebook base sigue sin ejecutarse (sus TODOs no están resueltos).
