import os
import sys
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# full db
#logs_path = '/root/logs_analysis/test_2021-03-06 20:30:23.730137'
# 200 samples
logs_path = '/root/logs_analysis/test_2021-03-09 09:46:43.297180'
# only variance - full db
logs_path = '/root/logs_analysis/test_2021-03-10 12:52:56.649979'
logs_path = '/root/logs_analysis/test_2021-03-10 14:10:39.179684'
out_path = '/root/logs_analysis_visualization_v2'


def sns_plot_variance_with_and_without_exp(variance_with_exp,
                                           variance_without_exp, fig_out_path,
                                           label1, label2):
    indexes = np.argsort(variance_without_exp)
    variance_without_exp = variance_without_exp[indexes]
    variance_with_exp = variance_with_exp[indexes]

    df = pd.DataFrame()
    df[label1] = variance_without_exp
    df[label2] = variance_with_exp

    plt.figure(figsize=(5, 5))
    sns.set()
    sns.set_style('white')
    sns.set_context('paper')
    sns.jointplot(data=df, x=label1, y=label2, kind='hist')
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
        df[str_] = all_errors[indexes, i]

        plt.figure(figsize=(10, 5))
        sns.set()
        sns.set_style('white')
        sns.set_context('paper')
        sns.violinplot(data=df, x='variance', y=str_)
        plt.title(title + src_names[i])
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
    plt.xlabel('pixels')
    plt.savefig(fig_out_path)
    plt.close()


def process_split(channel, variance_path, w_variance_path, errors_paths,
                  all_metrics_paths, metrics, split_name, gt_type, out_path):
    """
    variance_path - path of csv containing variances 
    w_variance_path - path of csv containing weighted variances - according to an ensemble function
    errors_paths - paths of edges errors 
    split_name - name of the split set 
    gt_type - which gt was considered - "twd_exp"/"twd_gt"

    """
    ##### valid data ######
    df = pd.read_csv(variance_path)
    df = df[df['channel'] == channel]
    variance_with_exp = df['variance_with_exp'].values
    variance_without_exp = df['variance_without_exp'].values

    h_var_with_exp, _ = np.histogram(variance_with_exp, bins=1000)
    plt.plot(h_var_with_exp)
    plt.savefig('ch_%d_%s_var_with_exp_1000bins.png' % (channel, split_name))
    plt.close()
    print(
        'ch %d %s %s var_with_exp -- min %20.10f -- max %20.10f -- median %20.10f'
        % (channel, split_name, gt_type, np.min(variance_with_exp),
           np.max(variance_with_exp), np.median(variance_with_exp)))

    h_var_with_exp, _ = np.histogram(variance_without_exp, bins=1000)
    plt.plot(h_var_with_exp)
    plt.savefig('ch_%d_%s_var_without_exp_1000bins.png' %
                (channel, split_name))
    plt.close()
    print(
        'ch %d %s %s var_without_exp -- min %20.10f -- max %20.10f -- median %20.10f'
        % (channel, split_name, gt_type, np.min(variance_without_exp),
           np.max(variance_without_exp), np.median(variance_without_exp)))
    '''
    min_v = min(np.min(variance_with_exp), np.min(variance_without_exp))
    max_v = max(np.max(variance_with_exp), np.max(variance_without_exp))
    variance_with_exp = (variance_with_exp - min_v) / (max_v - min_v)
    variance_without_exp = (variance_without_exp - min_v) / (max_v - min_v)

    import pdb
    pdb.set_trace()
    '''
    '''
    w_df = pd.read_csv(w_variance_path)
    w_df = w_df[w_df['channel'] == channel]
    w_variance_with_exp = w_df['variance'].values
    min_v = np.min(w_variance_with_exp)
    max_v = np.max(w_variance_with_exp)
    w_variance_with_exp = (w_variance_with_exp - min_v) / (max_v - min_v)

    set_str = '%s_set_%s_check' % (split_name, gt_type)
    print('process ' + set_str)
    print('process variances')
    ### plot variance with exp vs variance without exp
    #sns_plot_variance_with_and_without_exp(
    #    variance_with_exp, variance_without_exp,
    #    os.path.join(out_path,
    #                 '%s_variance_with_vs_without_exp.png' % (set_str)))

    sns_plot_variance_with_and_without_exp(
        variance_with_exp, w_variance_with_exp,
        os.path.join(out_path, '%s_variance_vs_w_variance.png' % (set_str)),
        'variance', 'weighted variance')
    print('process errors')

    ### get errors
    src_names = []
    all_errors = []
    prefix = 'errors_%s_%s_' % (split_name, gt_type[4:])
    for path in errors_paths:
        df_errors = pd.read_csv(path)
        df_errors = df_errors[df_errors['channel'] == channel]
        src_name = os.path.split(path)[-1][len(prefix):-4]
        src_names.append(src_name)
        all_errors.append(df_errors['errors'].values)
    import pdb
    pdb.set_trace()
    #sns_plot_variance_vs_indiv_errors(
    #    variance_without_exp, all_errors, src_names,
    #    '%s - variance (without exp) vs. edge errors ' % set_str,
    #    os.path.join(out_path,
    #                 '%s_variances_without_exp_vs_edge_errors.png' % set_str))

    #sns_plot_variance_vs_indiv_errors(
    #    variance_with_exp, all_errors, src_names,
    #    '%s - variance (with exp) vs. edge errors ' % set_str,
    #    os.path.join(out_path,
    #                 '%s_variances_with_exp_vs_edge_errors.png' % set_str))

    #sns_plot_variance_vs_indiv_errors(
    #    w_variance_with_exp, all_errors, src_names,
    #    '%s - w variance (with exp) vs. edge errors ' % set_str,
    #    os.path.join(out_path,
    #                 '%s_w_variances_with_exp_vs_edge_errors.png' % set_str))

    #plot_variance_and_errors(
    #    variance_without_exp, all_errors,
    #    os.path.join(
    #        out_path, '%s_variances_without_exp_vs_avg_errors_prev.png' %(set_str)))
    #
    #sns_plot_variance_vs_avg_errors(
    #    variance_without_exp, all_errors, src_names,
    #    '%s - Variance (without exp) vs. avg edge errors ' % set_str,
    #    os.path.join(out_path,
    #                 '%s_variances_without_exp_vs_avg_errors.png' % set_str))

    sns_plot_variance_vs_avg_errors(
        variance_with_exp, all_errors, src_names,
        '%s - Variance (with exp) vs. avg edge errors ' % set_str,
        os.path.join(out_path,
                     '%s_variances_with_exp_vs_avg_errors.png' % set_str))
    sns_plot_variance_vs_avg_errors(
        w_variance_with_exp, all_errors, src_names,
        '%s - w variance (with exp) vs. avg edge errors ' % set_str,
        os.path.join(out_path,
                     '%s_w_variances_with_exp_vs_avg_errors.png' % set_str))
    '''
    '''
    ### get metrics
    for idx in range(len(metrics)):
        metric_name = metrics[idx]
        metrics_paths = all_metrics_paths[idx]
        print('process metric ' + metric_name)
        src_names = []
        all_errors = []
        prefix = 'score_%s_%s_%s_' % (metric_name, split_name, gt_type[4:])
        for path in metrics_paths:
            df_errors = pd.read_csv(path)
            df_errors = df_errors[df_errors['channel'] == channel]
            src_name = os.path.split(path)[-1][len(prefix):-4]
            src_names.append(src_name)
            all_errors.append(df_errors['errors'].values)

        sns_plot_variance_vs_indiv_errors(
            variance_without_exp, all_errors, src_names,
            '%s - variance (without exp) vs. edge %s metric ' %
            (set_str, metric_name),
            os.path.join(
                out_path, '%s_variances_without_exp_vs_edge_%s_metric.png' %
                (set_str, metric_name)))

        sns_plot_variance_vs_avg_errors(
            variance_without_exp, all_errors, src_names,
            '%s - Variance (without exp) vs. avg edge %s metric ' %
            (set_str, metric_name),
            os.path.join(
                out_path,
                '%s_variances_without_exp_vs_avg_edge_%s_metric.png' %
                (set_str, metric_name)))
    '''


dst_tasks = os.listdir(logs_path)
dst_tasks.sort()
metrics = ['l1', 'l2', 'ssim', 'lpips']
for dst_task in dst_tasks:
    dst_task_out_path = os.path.join(out_path, dst_task)
    # prepare output path - where we store images
    os.makedirs(dst_task_out_path, exist_ok=True)

    variance_valid_path = '%s/%s/variance_valid.csv' % (logs_path, dst_task)
    variance_test_path = '%s/%s/variance_test.csv' % (logs_path, dst_task)
    w_variance_valid_path = '%s/%s/w_variance_valid_0.csv' % (logs_path,
                                                              dst_task)
    w_variance_test_path = '%s/%s/w_variance_test_0.csv' % (logs_path,
                                                            dst_task)
    errors_val_exp_paths = glob.glob('%s/%s/errors_valid_exp_*' %
                                     (logs_path, dst_task))
    errors_test_exp_paths = glob.glob('%s/%s/errors_test_exp_*' %
                                      (logs_path, dst_task))
    errors_test_gt_paths = glob.glob('%s/%s/errors_test_gt_*' %
                                     (logs_path, dst_task))
    all_metrics_val_exp_paths = []
    all_metrics_test_exp_paths = []
    all_metrics_test_gt_paths = []
    for metric_name in metrics:
        all_metrics_val_exp_paths.append(
            glob.glob('%s/%s/score_%s_valid_exp_*' %
                      (logs_path, dst_task, metric_name)))
        all_metrics_test_exp_paths.append(
            glob.glob('%s/%s/score_%s_test_exp_*' %
                      (logs_path, dst_task, metric_name)))
        all_metrics_test_gt_paths.append(
            glob.glob('%s/%s/score_%s_test_gt_*' %
                      (logs_path, dst_task, metric_name)))

    # get nr channels per dst domain
    df = pd.read_csv(variance_valid_path)
    n_channels = df['channel'].max() + 1

    for channel in range(n_channels):

        process_split(channel, variance_valid_path, w_variance_valid_path,
                      errors_val_exp_paths, all_metrics_val_exp_paths, metrics,
                      'valid', 'twd_exp', dst_task_out_path)
        process_split(channel, variance_test_path, w_variance_test_path,
                      errors_test_exp_paths, all_metrics_test_exp_paths,
                      metrics, 'test', 'twd_exp', dst_task_out_path)
        process_split(channel, variance_test_path, w_variance_test_path,
                      errors_test_gt_paths, all_metrics_test_gt_paths, metrics,
                      'test', 'twd_gt', dst_task_out_path)
