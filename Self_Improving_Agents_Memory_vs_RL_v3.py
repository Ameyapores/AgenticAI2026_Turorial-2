import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from simulator import ACTIONS, HORIZON, SyntheticProcess, train_q_learning, run_episode, evaluate_many, evaluate_custom_rule, report
from llm_client import connection_check, ask_json, ask_text

SEED = 7
TRAIN_EPISODES = 12_000
EVAL_SEEDS = range(10_000, 11_000)  # 1,000 never used for training

print('The LLM connection is required. If it fails, ask the tutorial organizers for configuration help.')
model_info = connection_check()

example_total, example_trace = run_episode('Fixed alternating', seed=2026)
print('One example of a fixed action schedule:')
display(pd.DataFrame(example_trace)[['step', 'before', 'action', 'after', 'gain', 'reward']])
print('Final score:', round(example_total, 2))

Q_reference, training_history = train_q_learning(TRAIN_EPISODES, seed=SEED)
policies = {
    'RL': Q_reference,
    'State-aware rule': None,
    'Fixed alternating': None,
    'Always STEADY': None,
    'Random': None,
}
score_arrays = evaluate_many(policies, seeds=EVAL_SEEDS)
reference_table = report(score_arrays)
display(reference_table)

paired_difference = score_arrays['RL'] - score_arrays['State-aware rule']
ci_half = 1.96 * paired_difference.std(ddof=1) / np.sqrt(len(paired_difference))
print(f'RL vs state-aware rule: +{paired_difference.mean():.2f} reward points')
print(f'Approximate paired 95% CI: [{paired_difference.mean()-ci_half:.2f}, {paired_difference.mean()+ci_half:.2f}]')

order = ['Random','Always STEADY','Fixed alternating','State-aware rule','RL']
means = [score_arrays[k].mean() for k in order]
colors = ['#a3b0c3','#a3b0c3','#1e3869','#1e3869','#bd7c49']
fig, ax = plt.subplots(figsize=(9.5,4.5))
ax.barh(order, means, color=colors)
ax.invert_yaxis()
for i,mean in enumerate(means): ax.text(mean+1, i, f'{mean:.1f}', va='center', weight='bold')
ax.set_xlim(0, 90)
ax.set_xlabel('Mean total reward across 1,000 unseen experiments')
ax.set_title('How much better is the learned policy?')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.show()

# Participant exercise: edit only the function below, then run this section.
# Return exactly one action label: "REST", "STEADY", or "BOOST".
def participant_rule(state):
    step, nutrient, stress = state
    if nutrient >= 3 and stress == 0:
        return "BOOST"
    if nutrient == 0 or stress >= 2:
        return "REST"
    return "STEADY"

participant_scores = evaluate_custom_rule(participant_rule, seeds=EVAL_SEEDS)
participant_comparison = report({
    'Participant rule': participant_scores,
    'Reference state-aware rule': score_arrays['State-aware rule'],
    'RL reference': score_arrays['RL'],
})
display(participant_comparison)

participant_total, participant_trace = run_episode(
    'Custom rule', seed=2026, custom_rule=participant_rule
)
print('Participant-rule score on example seed:', round(participant_total, 2))
display(pd.DataFrame(participant_trace)[['step', 'before', 'action', 'after', 'gain', 'reward']])

TRY_TRAIN_EPISODES = 100  # CHANGE ME: try 500, 3_000, 12_000

Q_changed, _changed_training = train_q_learning(TRY_TRAIN_EPISODES, seed=SEED)
changed = evaluate_many({'RL':Q_changed}, seeds=EVAL_SEEDS)['RL']
comparison = pd.DataFrame({
    'policy': [f'RL after {TRY_TRAIN_EPISODES:,} runs',
               f'RL after {TRAIN_EPISODES:,} runs',
               'State-aware handwritten rule'],
    'mean_reward': [changed.mean(), score_arrays['RL'].mean(),
                    score_arrays['State-aware rule'].mean()],
})
display(comparison.round(2))
plt.figure(figsize=(8,3.5))
plt.barh(comparison.policy, comparison.mean_reward, color=['#9ca9bd','#bc7d4b','#1e3869'])
plt.gca().invert_yaxis(); plt.xlabel('Mean held-out eight-step reward')
plt.xlim(0,90);plt.title('What changed when the controller got more experience?')
plt.tight_layout();plt.show()

CHECKPOINTS = [100,500,3000,12000]
TRAINING_SEEDS = [7,11,23,31,42]
progress_rows=[]
for episodes in CHECKPOINTS:
    for training_seed in TRAINING_SEEDS:
        q,_ = train_q_learning(episodes, seed=training_seed)
        # Separate held-out evaluation seeds from the reference run.
        outcome=evaluate_many({'RL':q},seeds=range(12000,12500))['RL']
        progress_rows.append({'training_episodes':episodes,
                              'training_seed':training_seed,
                              'heldout_mean':outcome.mean()})
progress=pd.DataFrame(progress_rows)
progress_summary=progress.groupby('training_episodes').heldout_mean.agg(['mean','std'])
display(progress_summary.round(2))

x=np.arange(len(CHECKPOINTS))
fig,ax=plt.subplots(figsize=(8,4))
ax.errorbar(x,progress_summary['mean'],yerr=progress_summary['std'],color='#bd7c49',
            fmt='o-',capsize=4,lw=2,label='RL mean ± SD across 5 training seeds')
ax.axhline(score_arrays['State-aware rule'].mean(),ls='--',lw=2,color='#1e3869',label='State-aware rule')
ax.set_xticks(x,[f'{v:,}' for v in CHECKPOINTS]);ax.set_xlabel('Training campaigns')
ax.set_ylabel('Held-out mean reward');ax.set_ylim(50,83)
ax.set_title('More experience improves the learned policy')
ax.legend();plt.tight_layout();plt.show()

HISTORICAL_PROTOCOLS = [
    ('Always STEADY',301),('Fixed alternating',302),('State-aware rule',303),
    ('Fixed alternating',304),('State-aware rule',305),('Always STEADY',306),
]
experience_rows=[]
for name,seed in HISTORICAL_PROTOCOLS:
    total,trace=run_episode(name,seed=seed)
    experience_rows.append({
        'protocol':name, 'seed':seed,'total_reward':round(total,2),
        'decisions':[r['action'] for r in trace],
        'observed_outcomes':[(r['before'],r['action'],r['gain'],r['after']) for r in trace]
    })
display(pd.DataFrame([{k:v for k,v in r.items() if k!='observed_outcomes'}
                      for r in experience_rows]))

memory_prompt=f"""You are creating reusable memory for an LLM scientific decision agent.
The setting is a FICTIONAL eight-stage process. Each stage, choose REST, STEADY or BOOST.
REST restores resources and reduces stress. STEADY produces modest output and reduces stress;
BOOST can produce more but depletes resources and increases stress. The final objective
includes total production, costs, and a terminal stress penalty. Every step changes the next state.

Here are SIX prior simulation traces:
{json.dumps(experience_rows)}

Return 4 to 6 short, actionable lessons grounded in the traces, including when REST may help,
when BOOST may backfire and why time remaining matters. Clearly say that six episodes are insufficient
to establish an optimal policy. Do not invent numerical results."""
SAVED_MEMORY = ask_text(memory_prompt,max_tokens=900)
print('Saved memory produced by the LIVE reasoning model:\n')
print(SAVED_MEMORY)

LIVE_SEED = 2301  # Repeating this cell uses paid model calls.
ACTION_GUIDE = (
    "REST: gain 0; restore nutrient by 2; reduce stress by 2; cost 0.5. "
    "STEADY: if nutrient >= 1, usually gain 2; consume nutrient 1; reduce stress by 1; cost 0.7. "
    "BOOST: if nutrient >= 2, possibly gain 6; consume nutrient 2; increase stress by 2; cost 2.0. "
    "BOOST can fail at high stress. Occasional disturbances remove nutrient. "
    "Final objective = 4 * total production - costs - 5 * final stress. "
    "This is a synthetic, not a physical, experimental process."
)

def make_llm_controller(memory=None):
    def decide(state, trace):
        prompt = f'''You are controlling a SYNTHETIC scientific process for 8 decisions.
{ACTION_GUIDE}
Current state (step, nutrient, stress): {state}
Steps remaining: {8-state[0]}
Previous observations in THIS run: {json.dumps(trace)}
Saved lessons from earlier runs: {memory if memory else '(none)'}
Choose the next action to maximize the FINAL total reward rather than immediate gain.
Respond with JSON: {{"action":"REST|STEADY|BOOST","reason":"short explanation"}}.
Do not invent experiment observations.'''
        response = ask_json(prompt, max_tokens=1200)
        action = str(response.get('action','')).strip().upper()
        if action not in ACTIONS:
            raise ValueError(f'Unknown action returned by live LLM: {response!r}')
        return ACTIONS.index(action), str(response.get('reason',''))[:250]
    return decide

llm_results=[]
live_traces={}
for label, memory in [('LLM context only',None),('LLM with memory',SAVED_MEMORY)]:
    total,trace=run_episode('LLM',seed=LIVE_SEED,live_agent=make_llm_controller(memory))
    live_traces[label]=trace
    llm_results.append({'policy':label,'total_reward':round(total,2),
                        'actions':' / '.join(x['action'] for x in trace)})
for label,q in [('RL learned policy',Q_reference),('State-aware rule',None)]:
    policy='RL' if q is not None else label
    total,trace=run_episode(policy,seed=LIVE_SEED,q=q)
    live_traces[label]=trace
    llm_results.append({'policy':label,'total_reward':round(total,2),
                        'actions':' / '.join(x['action'] for x in trace)})
display(pd.DataFrame(llm_results))
print('This is ONE noisy paired example, not a statistical comparison of LLM and RL.')

# Look at decisions, not just scores.
display(pd.DataFrame(live_traces['LLM with memory'])[
    ['step','before','action','after','gain','reward','why']])
