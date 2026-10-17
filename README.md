# U of T Fall School | Self-Improving Scientific Agents, Lab v3

## The question

When does training a decision policy improve performance relative to a fixed workflow or a reasoning model used for planning?

This repository contains a synthetic eight-step resource-and-stress control experiment. It is not a validated physical or biological model. The lab compares:

1. Random interventions.
2. Always STEADY, a fixed protocol.
3. Fixed alternating BOOST / REST, another fixed protocol.
4. A state-aware handwritten rule that reacts to measured state.
5. An RL-trained controller, with a separate tabular Q-policy and unchanged LLM weights.
6. An LLM with current-context reasoning, compared with the same LLM supplied with lessons from six prior campaigns.

The first five methods are benchmarked over 1,000 held-out simulated campaigns. The LLM demonstration uses one paired seed and cannot establish statistical superiority. The RL controller is trained on substantially more simulated experience than the LLM-memory condition.

## 1. Install software

Install Python 3.11 or 3.12, unzip the participant package, and open a terminal in the extracted folder.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyter lab
```

If your Python installation is 3.12, use `py -3.12` instead.

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter lab
```

Use the link displayed in the terminal to open JupyterLab in a web browser.

## 2. Configure the LLM API

Copy `api_config.py` to `api_config_local.py`, then paste your OpenRouter key into `API_KEY` in `api_config_local.py`. The provider, model ID, and endpoint are already set for OpenRouter. The local file is ignored by Git, and both the notebook and the standalone Python script read it.

The current settings are `PROVIDER = "openai_compatible"`, `MODEL = "openai/gpt-5.6-luna"`, and `BASE_URL = "https://openrouter.ai/api/v1"`. Change these only if you use a different provider or model.

Do not paste an API key in the notebook or include a real key in a commit. Before sharing or pushing changes, restore the placeholder value in `API_KEY`.

## 3. Open the notebook

In JupyterLab, open `Self_Improving_Agents_Memory_vs_RL_v3.ipynb`. Select the Python kernel from `.venv` (Jupyter may prompt you to install/register it). Start with **Run All**. The first code cell checks the school API with a real call; it will stop if the key or endpoint is invalid.

### Run it in VS Code

1. Install the Microsoft **Python** and **Jupyter** extensions in VS Code.
2. Use **File → Open Folder** and open the repository folder (the folder containing `simulator.py`, `llm_client.py`, and `api_config.py`). Do not open the notebook as an isolated file.
3. Open the integrated terminal. If `.venv` has not been created on this computer, run:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

4. Open `Self_Improving_Agents_Memory_vs_RL_v3.ipynb`. Choose **Select Kernel** in the upper right, then choose the Python environment at `.venv/bin/python`.
5. Choose **Clear All Outputs** in the notebook toolbar, then run the first code cell. It sends a real API request. If it succeeds, use **Run All** or run cells individually.

The notebook stores previous outputs. This copy may show an old `.env` configuration error until its outputs are cleared. In VS Code, select the local `.venv` kernel; a Jupyter Server password is not needed for that local kernel.

The parameter exercise is intended to be run before the LLM sections. Running all notebook cells also makes model calls in the final sections. One paired demonstration uses approximately 18 calls plus possible retries, subject to `LAB_MAX_MODEL_CALLS`.

## 4. The experiment you will run

Each campaign has eight decisions. At every step, the system has a **nutrient level** and **stress level**. You can REST to recover, use a STEADY intervention for smaller production, or BOOST for larger but riskier production. An occasional random disturbance removes nutrient. Actions affect future resources, future risks, and the final outcome.

Reward: **4 × total simulated production − total intervention costs − 5 × final stress**.

### Reference benchmark

Run Section 2. Its expected result (seed 7; 1,000 held-out campaigns) is approximately:

| Controller | Mean simulated reward |
|---|---:|
| RL, 12,000 training campaigns | **75.57** |
| State-aware handwritten rule | 67.48 |
| Fixed BOOST / REST alternation | 64.83 |
| Random | 28.97 |
| Always STEADY | 14.25 |

The paired difference between RL and the state-aware rule is 8.09 reward points, with an approximate 95% interval of [7.11, 9.07]. This result does not establish performance relative to an optimal hand-coded policy or an LLM in other settings.

### Change one number

Section 3 contains:

```python
TRY_TRAIN_EPISODES = 100  # Try 500, 3_000, 12_000
```

Start with 100 and rerun that cell. Compare the trained controller to the *original 12,000-run controller* and the state-aware baseline. Then try 3,000 or 12,000. No extra LLM API calls are needed.

The following cell repeats training with **five seeds** so you can see whether the improvement is reproducible rather than a lucky single run.

### Memory and reasoning vs RL

Sections 4 and 5 require the school's live LLM API. They:

1. Collect six example trajectories from predefined protocols.
2. Ask the LLM to turn them into reusable lessons.
3. Run the same reasoning model once using only current context and once with saved memory.
4. Compare the outcomes with RL and a handwritten baseline on the same seeded task.

The LLM comparison consists of one run per condition and is not an estimate of expected LLM performance. The notebook has not been tested against the school's production endpoint.

## Instructional sequence

The following sequence is suitable for a 25- to 30-minute session.

| Segment | Suggested time | Activity |
|---|---:|---|
| Motivation | 3 min | Introduce resource depletion and stress as sequential state variables. |
| Reference benchmark | 5 min | Run Section 2 and examine the 1,000-seed comparison. |
| Parameter exercise | 5 min | Set `TRY_TRAIN_EPISODES=100`, compare it with 12,000, and optionally repeat at 3,000. |
| Learning curve | 3 min | Review the five-training-seed results and their dispersion. |
| Memory construction | 4 min | Use the API to summarize six prior trajectories as reusable guidance. |
| LLM comparison | 6 min | Run context-only and memory-augmented conditions once, then inspect the action traces. |
| Discussion | 3 min | Discuss the assumptions required for repeated-interaction policy learning. |

## Reproducible benchmark results

Run `python run_benchmark.py` to reproduce the simulator-only analysis. It does not make API calls. With training seed 7, 12,000 training campaigns, and evaluation seeds 10,000–10,999, the results are:

| Controller | Mean return | SD across runs |
|---|---:|---:|
| RL learned policy | 75.57 | 12.30 |
| State-aware rule | 67.48 | 17.03 |
| Fixed BOOST/REST alternation | 64.83 | 23.32 |
| Random | 28.97 | 18.07 |
| Always STEADY | 14.25 | 7.35 |

The paired RL-minus-state-aware difference is 8.09 points, with an approximate 95% interval of [7.11, 9.07]. A shared seed fixes initial conditions but not the full stochastic path when policies take different actions.

The benchmark also trains policies using five independent training seeds and evaluates each on 500 held-out campaigns.

| Training campaigns | Mean RL return | SD across training seeds |
|---:|---:|---:|
| 100 | 62.66 | 3.46 |
| 500 | 68.56 | 1.68 |
| 3,000 | 73.90 | 1.29 |
| 12,000 | 75.95 | 0.49 |

Version 3 uses incremental production rewards and represents the state required for the delayed terminal stress penalty. This gives Q-learning a finite-horizon Markov process under the stated simulator. The results apply only to this implementation and its evaluation protocol.

## Files included

- `Self_Improving_Agents_Memory_vs_RL_v3.ipynb`: tutorial notebook.
- `Self_Improving_Agents_Memory_vs_RL_v3.py`: standalone Python version of the tutorial.
- `simulator.py`: synthetic process and RL learner.
- `llm_client.py`: school API interface with call limit.
- `api_config.py`: shareable API configuration template.
- `api_config_local.py`: ignored local configuration containing the API key; create it by copying `api_config.py`.
- `.gitignore`, `requirements.txt`: repository and package setup.
- `run_benchmark.py`: reproducibly regenerate all benchmark numbers with **no API calls**.
- `solutions/`: benchmark figures, CSVs, and JSON results for participant reference.

## Troubleshooting

**Authentication errors:** verify `LAB_LLM_PROVIDER`, `LAB_LLM_MODEL`, and the exact key/base URL in `.env`. Ask the organizer about rate limits.

**ModuleNotFoundError:** use the Jupyter Python kernel created by your `.venv`, and install all packages with that environment's Python.

**429 or timeouts:** wait or ask for help. Restarting `Run All` repeatedly will send extra paid calls.

**Model produces malformed JSON:** the helper may use a second live call to repair it; after the call cap it raises an error rather than silently substitute data.

**No RL benefit:** check that `TRAIN_EPISODES` is 12,000, the test uses the same reward settings as training, and the script was not accidentally modified. After changing environment dynamics, retrain the policy before evaluating on the altered environment.

## Scientific limitations

This synthetic experiment makes state-dependent sequential decisions relevant by construction. The results do not establish general superiority over carefully specified scripted controllers or LLMs. The memory condition receives six historical episodes, whereas RL receives thousands. Real scientific deployment requires independent validation, safety constraints, and an appropriate comparison of experience, computation, and API costs.
