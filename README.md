# U of T Fall School | Self-Improving Scientific Agents, Lab v3

## The question

**When do we gain something by training a decision policy, instead of using a fixed workflow or simply asking a reasoning model to plan?**

This is a completely synthetic eight-step resource-and-stress control experiment. It is **not a validated physical or biological model**. The classroom demo compares:

1. Random interventions.
2. Always STEADY, a fixed protocol.
3. Fixed alternating BOOST / REST, another fixed protocol.
4. State-aware handwritten rule, a stronger baseline that reacts to measured state.
5. An RL-trained controller, with a separate tabular Q-policy and unchanged LLM weights.
6. A live LLM with current-context reasoning, versus the same LLM with lessons from six past campaigns.

The first five methods are benchmarked over 1,000 held-out simulated campaigns. The live LLM demonstration uses one paired seed and **cannot establish statistical superiority**. The RL controller is trained on far more simulated experience than the LLM-memory demo has.

## 1. Install software

Install Python 3.11 or 3.12, unzip the participant package, and open a terminal in the extracted folder.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python.exe -m jupyter lab
```

If your Python installation is 3.12, use `py -3.12` instead.

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m jupyter lab
```

Use the link displayed in the terminal to open JupyterLab in a web browser.

## 2. Configure the school-provided LLM API

Before launching Jupyter, edit `.env` with the provider, exact model ID, API key and API URL your Fall School organizers provide. The `.env.example` file supports three routes:

- `LAB_LLM_PROVIDER=anthropic` using `ANTHROPIC_API_KEY`
- `LAB_LLM_PROVIDER=openai` using `OPENAI_API_KEY`
- `LAB_LLM_PROVIDER=openai_compatible` using `SCHOOL_LLM_API_KEY` and `SCHOOL_LLM_API_BASE`

The default `.env.example` includes placeholders, **not working credentials**. Do not paste an API key in the notebook, screenshot it, or commit `.env` to a Git repository.

## 3. Open the notebook

In JupyterLab, open `Self_Improving_Agents_Memory_vs_RL_v3.ipynb`. Select the Python kernel from `.venv` (Jupyter may prompt you to install/register it). Start with **Run All**. The first code cell checks the school API with a real call; it will stop if the key or endpoint is invalid.

**During the tutorial, the presenter will pause at the parameter exercise**. If you use Run All before class, the notebook will also make live model calls in its last sections. Expect approximately 18 calls plus retries, subject to `LAB_MAX_MODEL_CALLS`.

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

The paired difference between RL and the state-aware rule is **8.09 reward points**, approximate 95% CI **[7.11, 9.07]**. This does not imply RL would beat an optimal hand-coded policy or a strong LLM in general.

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

**Interpret the live LLM results cautiously**: one run per condition is a teaching demonstration, not an estimate of expected LLM performance. We have not verified the notebooks against the school's real endpoint.

## Files included

- `Self_Improving_Agents_Memory_vs_RL_v3.ipynb`: tutorial notebook.
- `Self_Improving_Agents_Memory_vs_RL_v3.py`: standalone Python version of the tutorial.
- `simulator.py`: synthetic process and RL learner.
- `llm_client.py`: school API interface with call limit.
- `.env.example`, `.gitignore`, `requirements.txt`: setup.
- `run_benchmark.py`: reproducibly regenerate all benchmark numbers with **no API calls**.
- `solutions/`: benchmark figures, CSVs, and JSON results for participant reference.
- `INSTRUCTOR_GUIDE.md`: teaching plan and scientific caveats.

## Troubleshooting

**Authentication errors:** verify `LAB_LLM_PROVIDER`, `LAB_LLM_MODEL`, and the exact key/base URL in `.env`. Ask the organizer about rate limits.

**ModuleNotFoundError:** use the Jupyter Python kernel created by your `.venv`, and install all packages with that environment's Python.

**429 or timeouts:** wait or ask for help. Restarting `Run All` repeatedly will send extra paid calls.

**Model produces malformed JSON:** the helper may use a second live call to repair it; after the call cap it raises an error rather than silently substitute data.

**No RL benefit:** check that `TRAIN_EPISODES` is 12,000, the test uses the same reward settings as training, and the script was not accidentally modified. After changing environment dynamics, retrain the policy before evaluating on the altered environment.

## Scientific limitations

This synthetic experiment was designed to make state-dependent, sequential decisions matter. The result is genuine for the stated setup, but it does **not** prove universal superiority over a carefully tuned scripted controller or a strong LLM. Model-memory conditions receive six historical episodes; RL receives thousands. Real scientific deployment requires independent experimental validation, instrument safety constraints, and an appropriate comparison of experience, computing and API costs.
