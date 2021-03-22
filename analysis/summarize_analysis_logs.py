import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd

logs_path = r'/data/multi-domain-graph-4/logs_analysis'
out_path = r'/data/multi-domain-graph-4/logs_summary'
os.makedirs(out_path, exist_ok=True)
'''
domain_name = 'depth'
formal_domain_name = 'depth_n_1_xtc'
#epochs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#epochs = np.array([50, 60, 70, 80, 90, 100])
#epochs = np.array([10, 20, 30, 40])
#epochs = np.array([50, 60, 70, 80])
#epochs = np.array([90])
epochs = np.array([100])
iter_idx = 2
#all_dfs = []
'''
domain_name = 'normals'
formal_domain_name = 'normals_xtc'
#epochs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#epochs = np.array([50, 60, 70, 80, 90, 100])
#epochs = np.array([10, 20, 30, 40])
#epochs = np.array([50, 60, 70, 80])
#epochs = np.array([90])
epochs = np.array([100])
iter_idx = 1
#all_dfs = []

for epoch_idx in epochs:
    logs_path_ = '%s/%s_iter%d_e%d/%s' % (logs_path, domain_name, iter_idx,
                                          epoch_idx, formal_domain_name)

    var_path = '%s/variance_test.csv' % (logs_path_)
    df_var = pd.read_csv(var_path)
    avg_var_with_exp = np.mean(df_var['variance_with_exp'].values)
    avg_var_without_exp = np.mean(df_var['variance_without_exp'].values)
    errors_paths = sorted(glob.glob('%s/errors_test_gt*' % logs_path_))

    idx = 0
    n_samples = 0
    all_errors = 0
    for err in errors_paths:
        df_err = pd.read_csv(err)
        if idx == 0:
            all_err = df_err['errors'].values
        else:
            all_err = all_err + df_err['errors'].values
        all_errors = all_errors + np.mean(df_err['errors'].values)
        idx = idx + 1
    avg_errors = np.mean(all_err / idx)
    avg_errors_v2 = all_errors / idx

    df = pd.DataFrame()
    df['variance_with_exp'] = [avg_var_with_exp]
    df['variance_without_exp'] = [avg_var_without_exp]
    df['errors'] = [avg_errors]
    df['errors_v2'] = [avg_errors_v2]
    df['epoch'] = [epoch_idx]
    df['iter_idx'] = [iter_idx]
    df['domain'] = [domain_name]
    #df = pd.concat(all_dfs)
    f = open(
        '%s/%s_iter%d_v2_e_%d.csv' %
        (out_path, domain_name, iter_idx, epoch_idx), 'w')
    f.write(df.to_csv(index=False))
    f.close()

    #all_dfs.append(df)

#df = pd.concat(all_dfs)
#f = open('%s/%s_iter%d_v2_e_%d.csv' % (out_path, domain_name, iter_idx, epoch_idx), 'w')
#f.write(df.to_csv(index=False))
#f.close()
