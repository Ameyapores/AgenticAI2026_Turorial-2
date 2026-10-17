# Instructor guide: v3 RL benefit demonstration

**Audience:** interdisciplinary postdocs with school-provided LLM APIs. No prior RL course required.

**Teaching objective:** demonstrate a case in which learning a policy from many repeated interactions achieves higher average reward than a plausible fixed protocol and a state-aware heuristic, while distinguishing that claim from broader superiority over an LLM.

## Talk sequence (25 to 30 minutes)

| Segment | Suggested time | Instructor action |
|---|---:|---|
| Motivation | 3 min | Explain why resource depletion and stress make an eight-step task sequential. |
| Reference | 5 min | Run Section 2. Show the 1,000-seed comparison and the 8.09 point advantage. |
| Change a parameter | 5 min | Set `TRY_TRAIN_EPISODES=100`; let students compare it with 12,000. Repeat at 3,000 if time. |
| Validate convergence | 3 min | Show the five-training-seed learning curve and explain its error shading. |
| Memory demonstration | 4 min | Generate reusable memory from six prior trajectories using the **live** API. |
| Compare live LLMs | 6 min | Run context-only and context+memory once, then inspect actions versus RL. |
| Discussion | 3 min | Why might RL help when there are many repeatable tasks? Where is it not justified? |

## Verified results from a genuine Python run

Training seed 7, 12,000 campaigns, 1,000 evaluation seeds 10,000–10,999:

- RL: 75.57 mean, SD 12.30
- State-aware rule: 67.48, SD 17.03
- Fixed BOOST/REST alternation: 64.83, SD 23.32
- Random: 28.97, SD 18.07
- Always STEADY: 14.25, SD 7.35

Paired RL minus state-aware difference: **8.09** points (approximate 95% interval [7.11, 9.07]). Shared seed controls the initial conditions but does not guarantee identical stochastic outcomes for different action sequences. This is a toy simulator, not a validated scientific system.

Independent learner-seed experiment, 500 held-out seeds per trained policy:

| Training campaigns | Mean RL return over 5 training seeds | SD across training seeds |
|---:|---:|---:|
| 100 | 62.66 | 3.46 |
| 500 | 68.56 | 1.68 |
| 3,000 | 73.90 | 1.29 |
| 12,000 | 75.95 | 0.49 |

These values come from `run_benchmark.py`, not an invented slide illustration.

## Why the old demonstration failed

The v2 synthetic lab compared RL against a **very strong scripted rule** and sometimes found RL losing. Additionally, the old state representation omitted accumulated reward-related state while delaying the whole production reward to the end, making the Bellman update harder to learn consistently. In v3, incremental production rewards are additive and the remaining delayed terminal stress penalty is a function of the represented state. Q-learning has a well-defined finite-horizon Markov process to learn.

## Important caveat when a student asks "Is RL better than Claude?"

**Not established.** The LLM with current state and previous trajectories can be an excellent decision-maker. This demo demonstrates two *different mechanisms*: language-based reflection plus memory, and a numerical policy trained on repeated interaction. RL's significant simulator advantage is shown against scripted controls. The live LLM comparison uses one paired seed and cannot substantiate a population-level advantage claim.

If comparing LLM vs RL systematically, obtain enough live inference runs, fix prompts, price token costs, normalize any external simulator interactions, and evaluate on held-out scientific tasks. Memory-only training includes six historical campaigns; Q-learning gets up to 12,000 synthetic campaigns, so sample exposure differs materially.

## API operations

Set `.env` using the school provider. Notebook Section 0 checks credentials live. Expect around 18 API calls for one paired live seed (one connection check, one memory summary, two policies each making eight decisions), with possible repair calls. Do not restart this section repeatedly without understanding cost and call caps. The live endpoint has not been verified by us.

## Slide synchronization

The PowerPoint v3 keeps the original v2 slides and inserts two validated experiment slides **immediately after its lab introduction**, explaining the 75.6 vs 67.5 score and the learning-curve checkpoints. All earlier v1/v2 files are unchanged.
