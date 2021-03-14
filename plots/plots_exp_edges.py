import os
import shutil
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = 'replica_exp_edges.csv'
fig_path = 'replica_exp_edges.png'

df = pd.read_csv(csv_path)

plt.figure(figsize=(10, 10))
sns.set()
sns.set_style('white')
sns.set_context('paper')
sns.lineplot(data=df, x='n_nodes', y='L1', hue='comb', style='selected')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(fig_path, bbox_inches='tight')
plt.close()