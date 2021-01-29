import os
import shutil
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#main_csvs_path = r'./tests_28.01'
#final_csv_path = r'tests_28.01.csv'
#fig_path = r'tests_28.01.svg'

#main_csvs_path = r'./csv_results_config1'
#final_csv_path = r'csv_results_config1.csv'
#fig_path = r'csv_results_config1.svg'

main_csvs_path = r'./csv_results_config6'
final_csv_path = r'csv_results_config6.csv'
fig_path = r'csv_results_config6.svg'


def generate_common_csv(main_csvs_path, final_csv_path):
    final_file = open(final_csv_path, 'w')
    all_csvs = os.listdir(main_csvs_path)
    all_csvs.sort()
    idx = 0
    for csv_name in all_csvs:
        csv_file = open(os.path.join(main_csvs_path, csv_name))
        lines = [line for line in csv_file]
        if idx > 0:
            lines = lines[1:]
        else:
            lines[0] = 'model,dataset,src_domain,dst_domain,l1,\n'
        for line in lines:
            final_file.write(line)
        csv_file.close()
        idx = idx + 1

    final_file.close()


def generate_plots(final_csv_path, fig_path):
    df = pd.read_csv(final_csv_path)
    df.drop_duplicates()
    all_datasets = df['dataset'].unique()
    for dataset in all_datasets:
        df_dataset = df.loc[df['dataset'] == dataset]
        db_fig_path = fig_path.replace('.svg', '_%s.svg' % dataset)

        plt.figure(figsize=(10, 10))
        sns.set()
        sns.set_style('white')
        sns.set_context('paper')
        sns.lineplot(x='model',
                     y='l1',
                     data=df_dataset,
                     style='src_domain',
                     hue='dst_domain')
        sns.scatterplot(x='model',
                        y='l1',
                        data=df_dataset,
                        style='src_domain',
                        hue='dst_domain')

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(db_fig_path, bbox_inches='tight')
        plt.close()


def generate_train_plots(csv_path, fig_path):
    df = pd.read_csv(csv_path)
    df.drop_duplicates()
    import pdb
    pdb.set_trace()
    all_src_domains = df['src_domain'].unique()
    for src_domain in all_src_domains:
        df_src = df.loc[df['src_domain'] == src_domain]
        src_fig_path = fig_path.replace('.svg', '_from_%s.svg' % src_domain)
        plt.figure(figsize=(10, 10))
        sns.set()
        sns.set_style('white')
        sns.set_context('paper')
        sns.lineplot(x='model',
                     y='l1',
                     data=df_src,
                     hue='config',
                     style='config')
        sns.scatterplot(x='model',
                        y='l1',
                        data=df_src,
                        hue='config',
                        style='config')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(src_fig_path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    generate_common_csv(main_csvs_path, final_csv_path)
    #generate_plots(final_csv_path, fig_path)

    #trains_csv_path = 'train_to_normals_28_01.csv'
    #out_path = 'train_to_normals_28_01.svg'
    #generate_train_plots(trains_csv_path, out_path)