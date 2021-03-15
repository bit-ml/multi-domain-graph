import os
import shutil
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = 'replica_edges_analysis.csv'
fig_path = 'replica_exp_edges.svg'

colors = ['#488f31', '#de425b']

df = pd.read_csv(csv_path)
df = df.replace({'selected': 0}, 'random')
df = df.replace({'selected': 1}, 'performance-based')
df = df.replace({'comb': 'our_median'}, 'CShift')
df = df.replace({'comb': 'simple_mean'}, 'Ensemble Mean')
df = df[df['comb'] != 'our_mean']
df = df.rename(columns={
    "comb": "Ensemble Method",
    "selected": "Node selection"
})

df['n_nodes'] = df['n_nodes'] + 1
domains = ['depth', 'normals']

fig, ax = plt.subplots(len(domains),
                       figsize=(6, 5 * len(domains)),
                       sharex=False)
fig.suptitle(
    'Performance evolution \n under different node selection strategies',
    fontsize=15,
    fontweight='bold')
sns.set()
sns.set_style('white')
sns.set_context('paper')

for i in range(len(domains)):
    df_dom = df[df['dst_node'] == domains[i]]

    ax[i] = sns.lineplot(data=df_dom,
                         x='n_nodes',
                         y='L1',
                         hue='Ensemble Method',
                         style='Node selection',
                         ax=ax[i],
                         linewidth=2,
                         palette=colors)
    if i < len(domains) - 1:
        ax[i].get_legend().remove()
    else:
        handles, labels = ax[i].get_legend_handles_labels()
        ax[i].get_legend().remove()
        leg = ax[i].legend(handles,
                           labels,
                           ncol=2,
                           loc='center',
                           bbox_to_anchor=(0.5, -0.35),
                           frameon=True,
                           handlelength=2.5,
                           fontsize=12.5)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)
    ax[i].tick_params(axis='x', labelsize=12.5)
    ax[i].tick_params(axis='y', labelsize=12.5)
    ax[i].set_ylabel('%s L1' % domains[i], size=15)
    ax[i].set_xlabel('number of nodes / tasks in the Multi-Task Graph',
                     fontsize=15)

plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()
'''
colors1 = ['#ffa600', '#f95d6a']
colors2 = ['#003f5c', '#665191']

fig, ax = plt.subplots(1, figsize=(6, 5), sharex=False)
df_depth = df[df['dst_node'] == 'depth']
df_normals = df[df['dst_node'] == 'normals']
sns.lineplot(data=df_depth,
             x="n_nodes",
             y="L1",
             hue='Ensemble Method',
             style='Node selection',
             legend=True,
             ax=ax,
             palette=colors1)
ax.set_ylabel('depth L1', size=15)
ax.set_xlabel('number of nodes / tasks in the Multi-Task Graph', fontsize=15)
ax2 = ax.twinx()
sns.lineplot(data=df_normals,
             x="n_nodes",
             y="L1",
             ax=ax2,
             hue='Ensemble Method',
             style='Node selection',
             legend=True,
             palette=colors2)
ax2.set_ylabel('normals L1', size=15)
ax.figure.legend()
plt.savefig(fig_path.replace('.png', '_.png'), bbox_inches='tight', dpi=300)
plt.close()
'''