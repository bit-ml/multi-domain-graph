import os
import sys
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# full db
logs_path = '/root/logs_analysis/test_2021-03-06 12:01:23.021828'
# 200 samples
#logs_path = '/root/logs_analysis/test_2021-03-06 13:35:21.210468'
# 10 samples
#logs_path = '/root/logs_analysis/test_2021-03-06 10:36:05.195071'
out_path = '/root/logs_analysis_visualization'


def sns_plot_variance_with_and_without_exp(variance_with_exp,
                                           variance_without_exp, fig_out_path):
    indexes = np.argsort(variance_without_exp)
    variance_without_exp = variance_without_exp[indexes]
    variance_with_exp = variance_with_exp[indexes]

    df = pd.DataFrame()
    df['variance considering only the edges'] = variance_without_exp
    df['variance considering both edges and expert'] = variance_with_exp

    plt.figure(figsize=(5, 5))
    sns.set()
    sns.set_style('white')
    sns.set_context('paper')
    sns.jointplot(data=df,
                  x='variance considering only the edges',
                  y='variance considering both edges and expert',
                  kind='hist')
    plt.legend()
    plt.savefig(fig_out_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_variance_with_and_without_exp(variance_with_exp, variance_without_exp,
                                       fig_out_path):
    indexes = np.argsort(variance_without_exp)
    variance_without_exp = variance_without_exp[indexes]
    variance_with_exp = variance_with_exp[indexes]

    plt.scatter(variance_without_exp, variance_with_exp, s=1)

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.ylabel('variance_with_exp')
    plt.xlabel('variance_without_exp')
    #plt.legend()
    plt.savefig(fig_out_path)
    plt.close()


def sns_plot_variance_vs_indiv_errors(variance, all_errors, src_names, title,
                                      fig_out_path):
    all_errors = np.stack(all_errors, 1)
    min_v = np.min(all_errors)
    max_v = np.max(all_errors)
    all_errors = (all_errors - min_v) / (max_v - min_v)

    indexes = np.argsort(variance)
    variance = variance[indexes]
    _, hist_bins = np.histogram(variance, bins=100, range=(0, 1))
    dig = np.digitize(variance, hist_bins) - 1

    for i in range(len(src_names)):

        df = pd.DataFrame()
        df['variance'] = dig
        df['variance'] = df.variance.astype('category')
        str_ = 'error for src %s' % src_names[i]
        df['error for src %s' % src_names[i]] = all_errors[indexes, i]

        plt.figure(figsize=(10, 5))
        sns.set()
        sns.set_style('white')
        sns.set_context('paper')
        sns.violinplot(data=df, x='variance', y=str)
        plt.title(title + src_names[i])
        plt.legend()
        plt.xticks(rotation=90)
        plt.savefig(fig_out_path.replace('.png', '_%s.png' % (src_names[i])),
                    bbox_inches='tight',
                    dpi=300)
        plt.close()


def sns_plot_variance_vs_avg_errors(variance, all_errors, src_names, title,
                                    fig_out_path):
    all_errors = np.stack(all_errors, 1)
    min_v = np.min(all_errors)
    max_v = np.max(all_errors)
    all_errors = (all_errors - min_v) / (max_v - min_v)
    avg_errors = np.mean(all_errors, 1)

    indexes = np.argsort(variance)
    variance = variance[indexes]
    _, hist_bins = np.histogram(variance, bins=100, range=(0, 1))
    dig = np.digitize(variance, hist_bins) - 1
    df = pd.DataFrame()
    df['variance'] = dig
    df['variance'] = df.variance.astype('category')
    df['avg error'] = avg_errors[indexes]
    #for i in range(len(src_names)):
    #    df['error_src_%s' % src_names[i]] = all_errors[indexes, i]
    plt.figure(figsize=(10, 5))
    sns.set()
    sns.set_style('white')
    sns.set_context('paper')
    sns.violinplot(data=df, x='variance', y='avg error')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=90)
    plt.savefig(fig_out_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_variance_and_errors(variance, all_errors, fig_out_path):

    all_errors = np.stack(all_errors, 1)
    min_v = np.min(all_errors)
    max_v = np.max(all_errors)
    all_errors = (all_errors - min_v) / (max_v - min_v)
    avg_errors = np.mean(all_errors, 1)

    indexes = np.argsort(variance)
    variance = variance[indexes]
    avg_errors = avg_errors[indexes]

    histo_var, histo_bins = np.histogram(variance, bins=100, range=(0, 1))
    for bin_idx in range(100):
        indices_ = np.argwhere(variance >= histo_bins[bin_idx])
        indices__ = np.argwhere(variance < histo_bins[bin_idx + 1])
        indices = np.intersect1d(indices_, indices__)
        val = np.mean(avg_errors[indices])
        avg_errors[indices] = val

    plt.plot(variance, label='variance')
    plt.plot(avg_errors, label='average errors')
    plt.ylim(0, 1)
    #plt.legend()
    plt.xlabel('pixels')
    plt.savefig(fig_out_path)
    plt.close()


os.makedirs(out_path, exist_ok=True)

dst_tasks = os.listdir(logs_path)
dst_tasks.sort()

for dst_task in dst_tasks:
    variance_valid_path = '%s/%s/variance_valid.csv' % (logs_path, dst_task)
    variance_test_path = '%s/%s/variance_test.csv' % (logs_path, dst_task)
    errors_val_exp_paths = glob.glob('%s/%s/errors_valid_exp_*' %
                                     (logs_path, dst_task))
    errors_test_exp_paths = glob.glob('%s/%s/errors_test_exp_*' %
                                      (logs_path, dst_task))
    errors_test_gt_paths = glob.glob('%s/%s/errors_test_gt_*' %
                                     (logs_path, dst_task))
    ##### valid data ######
    df = pd.read_csv(variance_valid_path)
    variance_with_exp = df['variance_with_exp'].values
    variance_without_exp = df['variance_without_exp'].values
    min_v = min(np.min(variance_with_exp), np.min(variance_without_exp))
    max_v = max(np.max(variance_with_exp), np.max(variance_without_exp))
    variance_with_exp = (variance_with_exp - min_v) / (max_v - min_v)
    variance_without_exp = (variance_without_exp - min_v) / (max_v - min_v)

    #sns_plot_variance_with_and_without_exp(
    #    variance_with_exp, variance_without_exp,
    #    os.path.join(out_path, 'variances_valid_with_and_without_exp.png'))

    src_names = []
    all_errors = []
    prefix = 'errors_valid_exp_'
    for path in errors_val_exp_paths:
        df_errors = pd.read_csv(path)
        src_name = os.path.split(path)[-1][len(prefix):-4]
        src_names.append(src_name)
        all_errors.append(df_errors['errors'].values)

    sns_plot_variance_vs_indiv_errors(
        variance_without_exp, all_errors, src_names,
        'Variance (without exp) vs. edge errors (twd expert) - valid set - ',
        os.path.join(out_path,
                     'variances_valid_without_exp_vs_edge_errors_twd_exp.png'))
    #sns_plot_variance_vs_indiv_errors(
    #    variance_with_exp, all_errors, src_names,
    #    'Variance (with exp) vs. edge errors (twd expert) - valid set - ',
    #    os.path.join(out_path,
    #                 'variances_valid_with_exp_vs_edge_errors_twd_exp.png'))

    #plot_variance_and_errors(
    #    variance_without_exp, all_errors,
    #    os.path.join(
    #        out_path,
    #        'variances_valid_without_exp_vs_avg_errors_twd_exp_prev.png'))
    #sns_plot_variance_vs_avg_errors(
    #    variance_without_exp, all_errors, src_names,
    #    'Variance (without exp) vs. avg edge errors (twd expert)- valid set',
    #    os.path.join(out_path,
    #                 'variances_valid_without_exp_vs_avg_errors_twd_exp.png'))
    #sns_plot_variance_vs_avg_errors(
    #    variance_with_exp, all_errors, src_names,
    #    'Variance (with exp) vs. avg edge errors (twd expert) - valid set ',
    #    os.path.join(out_path,
    #                 'variances_valid_with_exp_vs_avg_errors_twd_exp.png'))

    #### test data ####
    df = pd.read_csv(variance_test_path)
    variance_with_exp = df['variance_with_exp'].values
    variance_without_exp = df['variance_without_exp'].values
    min_v = min(np.min(variance_with_exp), np.min(variance_without_exp))
    max_v = max(np.max(variance_with_exp), np.max(variance_without_exp))
    variance_with_exp = (variance_with_exp - min_v) / (max_v - min_v)
    variance_without_exp = (variance_without_exp - min_v) / (max_v - min_v)

    #sns_plot_variance_with_and_without_exp(
    #    variance_with_exp, variance_without_exp,
    #    os.path.join(out_path, 'variances_test_with_and_without_exp.png'))

    src_names = []
    all_errors = []
    prefix = 'errors_test_exp_'
    for path in errors_test_exp_paths:
        df_errors = pd.read_csv(path)
        src_name = os.path.split(path)[-1][len(prefix):-4]
        src_names.append(src_name)
        all_errors.append(df_errors['errors'].values)

    sns_plot_variance_vs_indiv_errors(
        variance_without_exp, all_errors, src_names,
        'Variance (without exp) vs. edge errors (twd expert) - test set - ',
        os.path.join(out_path,
                     'variances_test_without_exp_vs_edge_errors_twd_exp.png'))
    #sns_plot_variance_vs_indiv_errors(
    #    variance_with_exp, all_errors, src_names,
    #    'Variance (with exp) vs. edge errors (twd expert) - test set - ',
    #    os.path.join(out_path,
    #                 'variances_test_with_exp_vs_edge_errors_twd_exp.png'))

    #sns_plot_variance_vs_avg_errors(
    #    variance_without_exp, all_errors, src_names,
    #    'Variance (without exp) vs. avg edge errors (twd expert) - test set',
    #    os.path.join(out_path,
    #                 'variances_test_without_exp_vs_avg_errors_twd_exp.png'))
    #sns_plot_variance_vs_avg_errors(
    #    variance_with_exp, all_errors, src_names,
    #    'Variance (with exp) vs. avg edge errors (twd expert) - test set',
    #    os.path.join(out_path,
    #                 'variances_test_with_exp_vs_avg_errors_twd_exp.png'))

    src_names = []
    all_errors = []
    prefix = 'errors_test_gt_'
    for path in errors_test_gt_paths:
        df_errors = pd.read_csv(path)
        src_name = os.path.split(path)[-1][len(prefix):-4]
        src_names.append(src_name)
        all_errors.append(df_errors['errors'].values)

    sns_plot_variance_vs_indiv_errors(
        variance_without_exp, all_errors, src_names,
        'Variance (without exp) vs. edge errors (twd gt) - test set - ',
        os.path.join(out_path,
                     'variances_test_without_exp_vs_edge_errors_twd_gt.png'))
    #sns_plot_variance_vs_indiv_errors(
    #    variance_with_exp, all_errors, src_names,
    #    'Variance (with exp) vs. edge errors (twd gt) - test set - ',
    #    os.path.join(out_path,
    #                 'variances_test_with_exp_vs_edge_errors_twd_gt.png'))

    #sns_plot_variance_vs_avg_errors(
    #    variance_without_exp, all_errors, src_names,
    #    'Variance (without exp) vs. avg edge errors (twd gt) - test set',
    #    os.path.join(out_path,
    #                 'variances_test_without_exp_vs_avg_errors_twd_gt.png'))
    #sns_plot_variance_vs_avg_errors(
    #    variance_with_exp, all_errors, src_names,
    #    'Variance (with exp) vs. avg edge errors (twd gt) - test set',
    #    os.path.join(out_path,
    #                 'variances_test_with_exp_vs_avg_errors_twd_gt.png'))
