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
    'cartoon', 'sobel_s', 'sobel_m', 'sobel_l', 'superpixels', 'depth'
]

fig, ax = plt.subplots(len(domains),
                       figsize=(6, 5 * len(domains)),
                       sharex=False)
df = pd.read_csv(csv_path)
fig.suptitle('Performance evolution \n of individual edges',
             fontsize=15,
             fontweight='bold')
sns.set()
sns.set_style('white')
sns.set_context('paper')

for i in range(len(domains)):
    domain = domains[i]

    df_dom = df[df['dst_node'] == domain]
    all_sources = df_dom['src_node'].values
    new_all_sources = []
    for j in range(len(all_sources)):
        src_dom = all_sources[j]
        src_dom = new_domains[prev_domains.index(src_dom)]
        new_src = r'%s $\rightarrow$ %s' % (src_dom, domain)
        new_all_sources.append(new_src)
    df_dom['src_node'] = new_all_sources

    print(df_dom)

    ax[i] = sns.barplot(data=df_dom,
                        x='L1',
                        y='src_node',
                        hue='iteration',
                        palette=colors,
                        ax=ax[i])
    ax[i].tick_params(axis='x', labelsize=12.5)
    ax[i].tick_params(axis='y', labelsize=12.5)
    ax[i].set_xlabel('%s L1' % domains[i], size=15)
    ax[i].set_ylabel('', fontsize=15)

plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()