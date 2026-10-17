"""Reproduce all simulator-only numbers. No API key required for this script.

The lab notebook itself DOES require the school's real LLM API.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import train_q_learning, evaluate_many, report

HERE = Path(__file__).resolve().parent
NAVY = '#1d3768'
BRONZE = '#b97946'
SLATE = '#a0aabd'

Q, _history = train_q_learning(episodes=12000, seed=7)
policies={'RL learned policy':Q,'State-aware rule':None,
          'Fixed alternating':None,'Always STEADY':None,'Random':None}
# 'RL learned policy' is a display name, but the simulator expects policy='RL'.
raw = {}
for label,q in policies.items():
    name = 'RL' if q is not None else label
    raw[label] = evaluate_many({name:q}, seeds=range(10000,11000))[name]
summary = report(raw)
summary.to_csv(HERE/'benchmark_summary.csv',index=False)
print(summary.to_string(index=False))

adv = raw['RL learned policy']-raw['State-aware rule']
print(f'Paired advantage versus state-aware rule: {adv.mean():.2f} ± '
      f'{1.96*adv.std(ddof=1)/np.sqrt(len(adv)):.2f} (approximate 95% CI)')

fig,ax=plt.subplots(figsize=(9.5,5.2))
order=['Random','Always STEADY','Fixed alternating','State-aware rule','RL learned policy']
val=[raw[k].mean() for k in order]
colors=[SLATE,SLATE,NAVY,NAVY,BRONZE]
ax.barh(order,val,color=colors)
ax.invert_yaxis(); ax.set_xlabel('Mean total reward across 1,000 held-out campaigns')
ax.set_title('RL learns better decisions in this synthetic process')
for i,x in enumerate(val):ax.text(x+1,i,f'{x:.1f}',va='center',fontweight='bold')
ax.set_xlim(0,90);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(HERE/'benchmark_results.png',dpi=180);plt.close(fig)

checkpoints=[100,500,3000,12000]
train_seeds=[7,11,23,31,42]
rows=[]
for episodes in checkpoints:
    for tr_seed in train_seeds:
        q,_=train_q_learning(episodes,seed=tr_seed)
        rets = evaluate_many({'RL':q},seeds=range(12000,12500))['RL']
        rows.append({'training_episodes':episodes,'training_seed':tr_seed,
                     'heldout_mean':float(rets.mean())})
progress=pd.DataFrame(rows)
progress.to_csv(HERE/'learning_progress.csv',index=False)
print('\nAcross training seeds:')
print(progress.groupby('training_episodes').heldout_mean.agg(['mean','std']).round(2).to_string())

fig,ax=plt.subplots(figsize=(8.8,4.8))
by=progress.groupby('training_episodes').heldout_mean
means=by.mean().values
stds=by.std().values
x=np.arange(len(checkpoints))
ax.plot(x,means,'o-',lw=2.5,color=BRONZE,markersize=9,label='Learned policy')
ax.fill_between(x,means-stds,means+stds,color=BRONZE,alpha=.18,label='± 1 SD across training seeds')
base=raw['State-aware rule'].mean()
ax.axhline(base,linestyle='--',lw=2,color=NAVY,label=f'State-aware rule ({base:.1f})')
ax.set_xticks(x,[format(n,',') for n in checkpoints]);ax.set_xlabel('Training campaigns')
ax.set_ylabel('Mean total reward (held-out)');ax.set_ylim(50,82)
ax.set_title('Learning improves with experience')
ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right')
fig.tight_layout();fig.savefig(HERE/'learning_convergence.png',dpi=180);plt.close(fig)

results={
  'environment':'Synthetic 8-step resource and stress control process',
  'training_episodes':12000, 'training_seed':7,'evaluation_seeds':[10000,10999],
  'n_heldout':1000,
  'scores':{name:{'mean':float(vals.mean()),'sd':float(vals.std(ddof=1))} for name,vals in raw.items()},
  'paired_advantage_vs_state_aware':{'mean':float(adv.mean()),'approx_ci95_halfwidth':float(1.96*adv.std(ddof=1)/np.sqrt(len(adv)))},
  'learning_progress_by_training_seed':rows
}
(HERE/'benchmark_results.json').write_text(json.dumps(results,indent=2))
