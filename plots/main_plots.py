import os
import shutil
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

main_csvs_path = r'./trained_18.01.2021_got10k_4_frames_per_video'
final_csv_path = r'trained_18.01.2021_got10k_4_frames_per_video.csv'
fig_path = r'trained_18.01.2021_got10k_4_frames_per_video.svg'


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


if __name__ == "__main__":
    generate_common_csv(main_csvs_path, final_csv_path)
    generate_plots(final_csv_path, fig_path)