import os
import shutil
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = 'replica_indiv_edges.csv'
fig_path = 'replica_indiv_edges.svg'

colors = ['#de425b', '#488f31']
domains = ['depth', 'normals']

prev_domains = [
    'rgb', 'normals', 'edges', 'halftone', 'semantic seg.', 'grayscale', 'hsv',
    'cartoon', 'sobel_s', 'sobel_m', 'sobel_l', 'superpixels', 'depth'
]
new_domains = [
    'rgb', 'normals', 'edges', 'halftone', 'semantic seg.', 'grayscale', 'hsv',
    'cartoonization', 'small scale edges', 'medium scale edges',
    'large scale edges', 'super-pixel', 'depth'
]

domains_sorter = [
    'rgb', 'depth', 'normals', 'small scale edges', 'medium scale edges',
    'large scale edges', 'edges', 'semantic seg.', 'cartoonization',
    'super-pixel', 'halftone', 'grayscale', 'hsv'
]

df = pd.read_csv(csv_path)

dfs = []
for dom in domains:
    df_dom = df[df['dst_node'] == dom]

    df_it1 = df_dom[df_dom['iteration'] == 1]
    df_it2 = df_dom[df_dom['iteration'] == 2]

    l1_it1 = df_it1.L1.values
    l1_it2 = df_it2.L1.values
    rel_impro = 100 * (l1_it1 - l1_it2) / l1_it1
    all_sources = df_it1.src_node.values
    new_sources = []
    for j in range(len(all_sources)):
        src_dom = all_sources[j]
        src_dom = new_domains[prev_domains.index(src_dom)]
        new_sources.append(src_dom)

    new_df = pd.DataFrame()
    new_df['src_node'] = new_sources  #df_it1.src_node.values
    new_df['destination task'] = df_it1.dst_node.values
    new_df['L1'] = rel_impro
    new_df.src_node = new_df.src_node.astype("category")
    new_df.src_node.cat.set_categories(domains_sorter, inplace=True)
    new_df = new_df.sort_values(['src_node'])

    dfs.append(new_df)

df = pd.concat(dfs)

fig, ax = plt.subplots(1, figsize=(6, 5))
fig.suptitle('Relative improvement of individual edges\n between iterations',
             fontsize=15,
             fontweight='bold')
sns.set()
sns.set_style('white')
sns.set_context('paper')

sns.barplot(data=df,
            x='L1',
            y='src_node',
            hue='destination task',
            palette=colors)
plt.tick_params(axis='x', labelsize=12.5)
plt.tick_params(axis='y', labelsize=12.5)
plt.ylabel('source task', size=15)
plt.xlabel('relative L1 improvement (%)', size=15)

plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()