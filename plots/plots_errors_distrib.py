import os
import shutil
import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = '/data/multi-domain-graph-4/logs_summary/iter1_depth_l1_gt.csv'
out_path = 'iter1_depth_l1_gt.png'
# 0 rgb, 4 -depth, 5 - normals
index = 4
colors = sns.color_palette(n_colors=13)
final_colors = [(0, 0, 0)] + colors[0:index] + colors[index + 1:]


def plot_data(df, axis, title, legend):
    axis = sns.kdeplot(
        data=df,
        x='distances',
        hue='type',
        fill=True,
        alpha=0.5,
        #cumulative=True,
        common_norm=False,
        common_grid=True,
        #element="poly",
        #kind='kde',
        palette=final_colors,
        ax=axis)
    #ax=axis)
    '''
    if not legend:
        axis.get_legend().remove()
    else:
        handles, labels = axis.get_legend_handles_labels()
        axis.get_legend().remove()
        leg = axis.legend(handles,
                          labels,
                          ncol=2,
                          loc='center',
                          bbox_to_anchor=(0, -0.35),
                          frameon=True,
                          handlelength=2.5,
                          fontsize=17.5)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)
    '''
    axis.tick_params(axis='x', labelsize=15)
    axis.tick_params(axis='y', labelsize=15)
    #axis.set_yticks([])
    axis.set_xlabel('distances', size=17.5)
    axis.set_ylabel('density', fontsize=17.5)
    axis.set_title(title, size=17.5)
    axis.set_xlim(0, 1)


fig, axes = plt.subplots(nrows=2,
                         ncols=2,
                         figsize=(6 * 2, 6 * 2),
                         sharex=False,
                         sharey=True)
fig.subplots_adjust(hspace=0.4)
fig.subplots_adjust(wspace=0.4)
fig.suptitle('Distribution of per-pixel distances',
             fontsize=17.5,
             y=1,
             fontweight='bold')

sns.set()
sns.set_style('white')
sns.set_context('paper')
df = pd.read_csv(csv_path)
# twd exp, iter1
plot_data(df, axes[0, 0],
          'Iteration 1\nDistances towards the expert pseudo-labels', False)

# twd gt, iter1
plot_data(df, axes[0, 1], 'Iteration 1\nDistances towards the ground truth',
          False)

# twd exp, iter2
plot_data(df, axes[1, 0],
          'Iteration 2\nDistances towards the expert pseudo-labels', False)

# twd gt, iter2

plot_data(df, axes[1, 1], 'Iteration 2\nDistances towards the ground truth',
          True)

plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.close()
