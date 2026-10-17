"""An entirely SYNTHETIC, finite-horizon scientific-process control problem.

The process is a classroom illustration, not a validated model of any real
bioreactor, intervention, instrument, or scientific process. All outcomes are
simulation outputs. No physiological or laboratory recommendations are implied.

There are 8 decisions, three actions, a stochastic nutrient disturbance and an
end-of-campaign stress penalty. This is a true multi-step MDP because current
actions alter the resource and stress available at the next step.

The implementation trains an external tabular Q-policy. There is no LLM
fine-tuning, and all LLM calls live in the companion Jupyter notebook.
"""
from __future__ import annotations
import numpy as np

ACTIONS = ("REST", "STEADY", "BOOST")
HORIZON = 8

class SyntheticProcess:
    def __init__(self, seed: int = 0, horizon: int = HORIZON,
                 disturbance_prob: float = .10, final_stress_penalty: float = 5.):
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.disturbance_prob = disturbance_prob
        self.final_stress_penalty = final_stress_penalty
        self.reset()

    def reset(self):
        self.t = 0
        self.nutrient = int(self.rng.integers(2, 5))
        self.stress = int(self.rng.integers(0, 2))
        self.production = 0
        return self.state()

    def state(self):
        return (self.t, self.nutrient, self.stress)

    def step(self, action: int):
        if not (0 <= int(action) < 3) or self.t >= self.horizon:
            raise ValueError("Invalid action or episode has finished")
        action = int(action)
        n, s = self.nutrient, self.stress
        if action == 0:  # REST
            self.nutrient = min(4, n + 2)
            self.stress = max(0, s - 2)
            gained, cost = 0, .5
        elif action == 1:  # STEADY
            p_success = max(0., .88 - .06 * s) if n >= 1 else 0.
            gained = 2 if self.rng.random() < p_success else 0
            self.nutrient = max(0, n - 1)
            self.stress = max(0, s - 1)
            cost = .7
        else:  # BOOST
            p_success = max(.03, .94 - .24 * s) if n >= 2 else 0.
            gained = 6 if self.rng.random() < p_success else 0
            self.nutrient = max(0, n - 2)
            self.stress = min(3, s + 2)
            cost = 2.

        # Random external disturbance that is observable at the next decision.
        if self.rng.random() < self.disturbance_prob:
            self.nutrient = max(0, self.nutrient - 1)

        self.production += gained
        self.t += 1
        done = self.t == self.horizon

        # The 4 * gain signal is additive. The final stress penalty is delayed.
        # Keeping the reward Markov with respect to (t,nutrient,stress) matters.
        reward = 4. * gained - cost - (self.final_stress_penalty * self.stress if done else 0.)
        info = {"gain": gained, "cost": cost, "total_production": self.production,
                "terminal_stress_penalty": self.final_stress_penalty * self.stress if done else 0.}
        return self.state(), float(reward), done, info


def scripted_action(state, name):
    t, nutrient, stress = state
    if name == "Always STEADY":
        return 1
    if name == "Fixed alternating":
        return 2 if t % 2 == 0 else 0
    if name == "State-aware rule":
        if nutrient >= 2 and stress <= 1:
            return 2
        return 1 if nutrient >= 1 else 0
    if name == "Boost/rest heuristic":
        return 2 if nutrient >= 2 and stress <= 1 else 0
    raise ValueError("Unknown script: " + name)


def train_q_learning(episodes: int = 12000, seed: int = 7,
                     disturbance_prob: float = .10,
                     final_stress_penalty: float = 5.,
                     alpha_start: float = .35):
    """Online one-step tabular Q-learning with exploratory simulator rollouts.

    Future states are learned only through experience. The method is intentionally
    simple and cheap; the experiment is about *policy learning*, not algorithms.
    """
    rng = np.random.default_rng(seed)
    lab = SyntheticProcess(seed + 5, disturbance_prob=disturbance_prob,
                           final_stress_penalty=final_stress_penalty)
    q = np.zeros((HORIZON + 1, 5, 4, 3), dtype=np.float64)
    visits = np.zeros_like(q)
    history = np.empty(episodes, dtype=np.float64)

    for episode in range(episodes):
        state = lab.reset()
        eps = .035 + .40 * np.exp(-episode / 7500.)
        total = 0.
        for _ in range(HORIZON):
            if rng.random() < eps:
                action = int(rng.integers(3))
            else:
                # Random tie break on exact Q equality.
                qrow = q[state]
                action = int(rng.choice(np.flatnonzero(qrow == qrow.max())))
            next_state, reward, done, _info = lab.step(action)
            idx = state + (action,)
            visits[idx] += 1
            alpha = max(.04, alpha_start / ((1 + visits[idx] / 120.) ** .65))
            bootstrap = 0. if done else float(np.max(q[next_state]))
            q[idx] += alpha * (reward + bootstrap - q[idx])
            total += reward
            state = next_state
        history[episode] = total
    return q, history


def run_episode(policy: str, seed: int, q=None,
                disturbance_prob: float = .10,
                final_stress_penalty: float = 5.,
                live_agent=None):
    """Run and log one episode.

    A live_agent callable receives (state, prior_trace) and returns (action, reason).
    The same observations, action choices, and reward are exposed to each policy.
    """
    lab = SyntheticProcess(seed, disturbance_prob=disturbance_prob,
                           final_stress_penalty=final_stress_penalty)
    state = lab.state()
    trace = []
    total = 0.
    for _ in range(HORIZON):
        if policy == "RL":
            if q is None: raise ValueError("Pass Q for RL evaluation")
            action = int(np.argmax(q[state]))
            reason = "Learned decision policy"
        elif policy == "Random":
            action = int(lab.rng.integers(3))
            reason = "Random baseline"
        elif policy == "LLM":
            if live_agent is None: raise ValueError("Missing live LLM callback")
            action, reason = live_agent(state, trace)
        else:
            action = scripted_action(state, policy)
            reason = "Predefined decision rule"
        before = state
        state, reward, done, info = lab.step(action)
        total += reward
        trace.append({"step": len(trace) + 1, "before": before,
                      "action": ACTIONS[action], "after": state,
                      "gain": info['gain'], "reward": round(reward, 2),
                      "why": reason})
    return float(total), trace


def evaluate_many(policies, seeds=range(10000, 11000),
                  disturbance_prob=.10, final_stress_penalty=5.):
    """Return per-seed scores so uncertainty and paired differences are visible."""
    return {name: np.array([
        run_episode(name, seed=int(s), q=q, disturbance_prob=disturbance_prob,
                    final_stress_penalty=final_stress_penalty)[0]
        for s in seeds], dtype=float)
        for name, q in policies.items()}


def report(scores):
    import pandas as pd
    rows=[]
    for name, value in scores.items():
        rows.append({'policy':name, 'mean_return':round(float(value.mean()), 2),
                     'sd_across_runs':round(float(value.std(ddof=1)),2),
                     'se_mean':round(float(value.std(ddof=1)/np.sqrt(len(value))),2),
                     'n':len(value)})
    return pd.DataFrame(rows).sort_values('mean_return',ascending=False).reset_index(drop=True)
