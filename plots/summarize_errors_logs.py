import os
import shutil
import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
out_path = 'with_exp_prob_depth_l1.svg'
path_iter1 = '/data/multi-domain-graph-4/logs_analysis/iter1_depth_l1'
path_iter2 = '/data/multi-domain-graph-4/logs_analysis/iter2_depth_l1'
exp_errors_path = '/data/multi-domain-graph-4/logs_analysis/iter1_depth_exp/depth_n_1_xtc/ensemble_errors_test_exp_superpixel_fcn.csv'
domain_name = 'depth_n_1_xtc'
domain_name_str = 'depth'

x_max_lim = 0.8
y_max_lim_exp = 0.005
y_max_lim_gt = 0.003
#y_max_lim_iters_exp =
#y_max_lim_iters_gt =
'''
out_path = 'with_exp_prob_normals_l1.svg'
path_iter1 = '/data/multi-domain-graph-4/logs_analysis/iter1_normals_l1'
path_iter2 = '/data/multi-domain-graph-4/logs_analysis/iter2_normals_l1'
exp_errors_path = '/data/multi-domain-graph-4/logs_analysis/iter1_normals_exp/normals_xtc/ensemble_errors_test_exp_superpixel_fcn.csv'
domain_name = 'normals_xtc'
domain_name_str = 'normals'

x_max_lim = 1
y_max_lim_exp = 0.0015
y_max_lim_gt = 0.004

pattern_ensemble_gt = 'ensemble_errors_test_gt_*.csv'
pattern_ensemble_exp = 'ensemble_errors_test_exp_*.csv'
red_pattern_errors_gt = 'errors_test_gt'
red_pattern_errors_exp = 'errors_test_exp'
pattern_errors_gt = 'errors_test_gt_*.csv'
pattern_errors_exp = 'errors_test_exp_*.csv'

prev_domains = [
    'rgb', 'normals_xtc', 'edges_dexined', 'halftone_gray', 'sem_seg_hrnet',
    'grayscale', 'hsv', 'cartoon_wb', 'sobel_small', 'sobel_medium',
    'sobel_large', 'superpixel_fcn', 'depth_n_1_xtc'
]
new_domains = [
    'rgb', 'normals', 'edges', 'halftone', 'semantic seg.', 'grayscale', 'hsv',
    'cartoon', 'small edges', 'medium edges', 'large edges', 'super-pixel',
    'depth'
]

domains_sorter = [
    'rgb', 'halftone_gray', 'grayscale', 'hsv', 'depth_n_1_xtc', 'normals_xtc',
    'sobel_small', 'sobel_medium', 'sobel_large', 'edges_dexined',
    'superpixel_fcn', 'cartoon_wb', 'sem_seg_hrnet'
]
index = domains_sorter.index(domain_name)
domains_sorter.remove(domain_name)

colors = sns.color_palette(n_colors=13)
final_colors = [(0, 0, 0)] + colors[0:index] + colors[index + 1:]

ensemble_gt_path_iter1 = sorted(
    glob.glob('%s/%s/%s' % (path_iter1, domain_name, pattern_ensemble_gt)))[0]
ensemble_exp_path_iter1 = sorted(
    glob.glob('%s/%s/%s' % (path_iter1, domain_name, pattern_ensemble_exp)))[0]
errors_gt_paths_iter1 = sorted(
    glob.glob('%s/%s/%s' % (path_iter1, domain_name, pattern_errors_gt)))
errors_exp_paths_iter1 = sorted(
    glob.glob('%s/%s/%s' % (path_iter1, domain_name, pattern_errors_exp)))

ensemble_gt_path_iter2 = sorted(
    glob.glob('%s/%s/%s' % (path_iter2, domain_name, pattern_ensemble_gt)))[0]
ensemble_exp_path_iter2 = sorted(
    glob.glob('%s/%s/%s' % (path_iter2, domain_name, pattern_ensemble_exp)))[0]
errors_gt_paths_iter2 = sorted(
    glob.glob('%s/%s/%s' % (path_iter2, domain_name, pattern_errors_gt)))
errors_exp_paths_iter2 = sorted(
    glob.glob('%s/%s/%s' % (path_iter2, domain_name, pattern_errors_exp)))

errors_gt_paths_ = len(domains_sorter) * ['']
for path in errors_gt_paths_iter1:
    pos = path.find(red_pattern_errors_gt)
    src_dom = path[pos + len(red_pattern_errors_gt) + 1:-4]
    errors_gt_paths_[domains_sorter.index(src_dom)] = path

errors_exp_paths_ = len(domains_sorter) * ['']
for path in errors_exp_paths_iter1:
    pos = path.find(red_pattern_errors_exp)
    src_dom = path[pos + len(red_pattern_errors_exp) + 1:-4]
    errors_exp_paths_[domains_sorter.index(src_dom)] = path

errors_exp_paths_iter1 = errors_exp_paths_
errors_gt_paths_iter1 = errors_gt_paths_

errors_gt_paths_ = len(domains_sorter) * ['']
for path in errors_gt_paths_iter2:
    pos = path.find(red_pattern_errors_gt)
    src_dom = path[pos + len(red_pattern_errors_gt) + 1:-4]
    errors_gt_paths_[domains_sorter.index(src_dom)] = path

errors_exp_paths_ = len(domains_sorter) * ['']
for path in errors_exp_paths_iter2:
    pos = path.find(red_pattern_errors_exp)
    src_dom = path[pos + len(red_pattern_errors_exp) + 1:-4]
    errors_exp_paths_[domains_sorter.index(src_dom)] = path

errors_exp_paths_iter2 = errors_exp_paths_
errors_gt_paths_iter2 = errors_gt_paths_


def plot_data(df, axis, title, legend, x_max_lim, y_max_lim, final_colors):
    '''
    axis = sns.histplot(
        data=df,
        x='distances',
        hue='type',
        element="poly",
        #stat="count",
        kde=True,
        #kind='kde',
        palette=final_colors,
        ax=axis)
    '''

    axis = sns.kdeplot(
        data=df,
        x='distances',
        hue='type',
        #fill=True,
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
    axis.set_xlabel('per-pixel errors', size=17.5)
    #axis.set_ylabel('cumulative distribution', fontsize=17.5)
    axis.set_title(title, size=17.5)
    #axis.set_xlim(0, x_max_lim)
    #axis.set_ylim(0, y_max_lim)


def get_error_values(csv_path):

    df = pd.read_csv(csv_path)
    max_channel = np.max(df['channel'].values)
    if max_channel > 0:
        for ch in range(max_channel):
            df_ch = df[df['channel'] == ch]
            if ch == 0:
                values = df_ch['errors'].values
            else:
                values = values + df_ch['errors'].values
    else:
        values = df['errors'].values

    np.random.seed(115)
    indexes = np.arange(0, len(values))
    np.random.shuffle(indexes)
    indexes = indexes[0:65536]
    values = values[indexes]

    #values = values[0:100]

    return values


def get_df(ensemble_path, errors_paths, red_pattern, domain_name_str):
    all_dfs = []
    print('ens')
    values = get_error_values(ensemble_path)

    df = pd.DataFrame()
    df['distances'] = values
    df['type'] = 'CShift'
    all_dfs.append(df)

    for err_path in errors_paths:
        pos = err_path.find(red_pattern)
        src_dom = err_path[pos + len(red_pattern) + 1:-4]
        src_dom = new_domains[prev_domains.index(src_dom)]
        print(src_dom)
        values = get_error_values(err_path)
        df = pd.DataFrame()
        df['distances'] = values
        df['type'] = r'%s $\rightarrow$ %s' % (src_dom, domain_name_str)
        all_dfs.append(df)
    df = pd.concat(all_dfs)

    return df


def get_red_df(ensemble_path_iter1, ensemble_path_iter2, exp_errors_path,
               domain_name_str):
    all_dfs = []
    print('ens1')
    values = get_error_values(ensemble_path_iter1)

    df = pd.DataFrame()
    df['distances'] = values
    df['type'] = 'CShift - Iter 1'
    all_dfs.append(df)

    print('ens2')
    values = get_error_values(ensemble_path_iter2)

    df = pd.DataFrame()
    df['distances'] = values
    df['type'] = 'CShift - Iter 2'
    all_dfs.append(df)

    if not exp_errors_path == None:
        print('ens3')
        values = get_error_values(exp_errors_path)

        df = pd.DataFrame()
        df['distances'] = values
        df['type'] = 'Expert'
        all_dfs.append(df)

    df = pd.concat(all_dfs)

    return df


fig, axes = plt.subplots(nrows=3,
                         ncols=2,
                         figsize=(6 * 2, 6 * 3),
                         sharex=False)
fig.subplots_adjust(hspace=0.4)
fig.subplots_adjust(wspace=0.4)
fig.suptitle('Distribution of per-pixel distances\nfor %s' % domain_name_str,
             fontsize=17.5,
             y=1,
             fontweight='bold')

sns.set()
sns.set_style('white')
sns.set_context('paper')
#import pdb
#pdb.set_trace()

# twd exp, iter1
df = get_df(ensemble_exp_path_iter1, errors_exp_paths_iter1,
            red_pattern_errors_exp, domain_name_str)
plot_data(df, axes[0, 0],
          'Iteration 1\nDistances towards the expert pseudo-labels', False,
          x_max_lim, y_max_lim_exp, final_colors)

# twd gt, iter1
df = get_df(ensemble_gt_path_iter1, errors_gt_paths_iter1,
            red_pattern_errors_gt, domain_name_str)
plot_data(df, axes[0, 1], 'Iteration 1\nDistances towards the ground truth',
          False, x_max_lim, y_max_lim_gt, final_colors)

# twd exp, iter2
df = get_df(ensemble_exp_path_iter2, errors_exp_paths_iter2,
            red_pattern_errors_exp, domain_name_str)
plot_data(df, axes[1, 0],
          'Iteration 2\nDistances towards the expert pseudo-labels', True,
          x_max_lim, y_max_lim_exp, final_colors)

# twd gt, iter2
df = get_df(ensemble_gt_path_iter2, errors_gt_paths_iter2,
            red_pattern_errors_gt, domain_name_str)
plot_data(df, axes[1, 1], 'Iteration 2\nDistances towards the ground truth',
          False, x_max_lim, y_max_lim_gt, final_colors)

df = get_red_df(ensemble_exp_path_iter1, ensemble_exp_path_iter2, None,
                domain_name_str)
plot_data(
    df, axes[2, 0],
    'Iteration1 vs Iteration 2\nDistances towards the expert pseudo-labels',
    False, x_max_lim, y_max_lim_exp, [(0, 0, 1), (0, 1, 0)])

df = get_red_df(ensemble_gt_path_iter1, ensemble_gt_path_iter2,
                exp_errors_path, domain_name_str)
'''
v = df['distances'].values
for i in range(len(v)):
    if v[i].find('nan') > -1:
        v[i] = 0
df['distances'] = v
'''

plot_data(df, axes[2, 1],
          'Iteration1 vs Iteration 2\nDistances towards the ground truth',
          False, x_max_lim, y_max_lim_gt, [(0, 0, 1), (0, 1, 0), (1, 0, 0)])

plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.close()
