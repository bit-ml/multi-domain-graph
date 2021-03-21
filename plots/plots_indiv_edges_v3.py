import os
import shutil
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = 'replica_indiv_edges.csv'
fig_path = 'replica_indiv_edges.svg'

#colors = ['#de425b', '#488f31']
colors = ['#58508d', '#ff6361']
domains = ['depth', 'normals']

prev_domains = [
    'rgb', 'normals', 'edges', 'halftone', 'semantic seg.', 'grayscale', 'hsv',
    'cartoon', 'sobel_s', 'sobel_m', 'sobel_l', 'superpixels', 'depth'
]
new_domains = [
    'rgb', 'normals', 'edges', 'halftone', 'semantic seg.', 'grayscale', 'hsv',
    'cartoon', 'small edges', 'medium edges', 'large edges', 'super-pixel',
    'depth'
]

domains_sorter = [
    'rgb', 'halftone', 'grayscale', 'hsv', 'depth', 'normals', 'small edges',
    'medium edges', 'large edges', 'edges', 'super-pixel', 'cartoon',
    'semantic seg.'
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

fig, ax = plt.subplots(nrows=1,
                       ncols=len(domains),
                       figsize=(6 * len(domains), 5),
                       sharex=False)
fig.suptitle('Relative L1 improvement (%) of edges between iterations',
             fontsize=17.5,
             y=1,
             fontweight='bold')
sns.set()
sns.set_style('white')
sns.set_context('paper')

for i in range(len(domains)):
    domain = domains[i]

    df_dom = df[df['destination task'] == domain]
    '''
    all_sources = df_dom['src_node'].values
    new_all_sources = []
    for j in range(len(all_sources)):
        src_dom = all_sources[j]
        new_src = r'%s $\rightarrow$ %s' % (src_dom, domain)
        new_all_sources.append(new_src)
    df_dom['src_node'] = new_all_sources
    '''

    df_dom = df_dom[df_dom['src_node'] != domain]
    df_dom['src_node'] = df_dom['src_node'].astype(str)
    ax[i] = sns.barplot(data=df_dom,
                        x='src_node',
                        y='L1',
                        palette=[colors[i]],
                        ax=ax[i])
    ax[i].tick_params(axis='x', labelsize=15, rotation=90)
    ax[i].tick_params(axis='y', labelsize=15)
    #ax[i].set_xlabel('source node' % domains[i], size=15)
    ax[i].set_xlabel('', size=17.5)
    ax[i].set_ylabel('', fontsize=17.5)
    ax[i].set_title(r'Edges $\rightarrow %s$' % domain, size=17.5)

plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()