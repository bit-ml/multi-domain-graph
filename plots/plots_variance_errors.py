import os
import shutil
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = 'variance_logs.csv'
fig_path = 'variance_logs.svg'

#colors = ['#488f31', '#de425b']
#colors = ['#003f5c', '#ffa600']
colors = ['#58508d', '#ff6361']

df = pd.read_csv(csv_path)

df = df.rename(columns={"domain": "Destination node", "iter_idx": "Iteration"})
df['errors'] = df['errors'] * 100
df['Iteration'] = df['Iteration'].astype(str)

metrics = ['normalized variance', 'normalized average L1 scores']
datas = ['variance_without_exp', 'errors']

import pdb
pdb.set_trace()
dfs = []
df.loc[df['Destination node'] == 'normals', 'errors']
for dom in ['depth', 'normals']:
    df_d = df[df['Destination node'] == dom]

    err_values = df_d['errors'].values
    min_v = np.min(err_values)
    max_v = np.max(err_values)
    err_values = (err_values - min_v) / (max_v - min_v)

    v_values = df_d['variance_without_exp'].values
    min_v = np.min(v_values)
    max_v = np.max(v_values)
    v_values = (v_values - min_v) / (max_v - min_v)

    df.loc[df['Destination node'] == dom, 'errors'] = err_values
    df.loc[df['Destination node'] == dom, 'variance_without_exp'] = v_values
print(df)

fig, ax = plt.subplots(nrows=1,
                       ncols=len(metrics),
                       figsize=(6 * len(metrics), 5),
                       sharex=False)
fig.suptitle(r'Analysis of ensemble edges',
             fontsize=17.5,
             y=1,
             fontweight='bold')
sns.set()
sns.set_style('white')
sns.set_context('paper')

for i in range(len(metrics)):

    ax[i] = sns.lineplot(data=df,
                         x='epoch',
                         y=datas[i],
                         hue='Destination node',
                         style='Iteration',
                         ax=ax[i],
                         linewidth=3,
                         palette=colors)
    if i < len(metrics) - 1:
        ax[i].get_legend().remove()
    else:
        handles, labels = ax[i].get_legend_handles_labels()
        ax[i].get_legend().remove()
        leg = ax[i].legend(handles,
                           labels,
                           ncol=2,
                           loc='center',
                           bbox_to_anchor=(0, -0.35),
                           frameon=True,
                           handlelength=2.5,
                           fontsize=17.5)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)
    ax[i].tick_params(axis='x', labelsize=15)
    ax[i].tick_params(axis='y', labelsize=15)
    #ax[i].set_ylabel('%s L1' % domains[i], size=15)
    ax[i].set_ylabel('', size=17.5)
    ax[i].set_xlabel('epoch', fontsize=17.5)
    #ax[i].set_title('%s' % metrics[i], size=15)
    ax[i].set_title(metrics[i], size=17.5)
    #'Average per-pixel variance of edges \nreaching the same destination node',
    #size=15)

plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()
