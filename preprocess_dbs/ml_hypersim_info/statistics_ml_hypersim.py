import os
import shutil
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

csv_path = 'metadata_images_split_scene_v1.csv'
df = pd.read_csv(csv_path)
df_frames_included = df[df['included_in_public_release'] == True]

print("# initial frames %d -- # released frames %d" %
      (len(df), len(df_frames_included)))

df = df_frames_included
df_train = df[df['split_partition_name'] == 'train']
df_test = df[df['split_partition_name'] == 'test']
df_val = df[df['split_partition_name'] == 'val']

print('# train frames %d' % len(df_train))
print('# test frames %d' % len(df_test))
print('# val frames %d' % len(df_val))

print('# scenes %d' % len(df['scene_name'].unique()))
print('max nr cameras per scene %d' % (len(df['camera_name'].unique())))
print('max frames per camera %d' % (np.max(df['frame_id'])))

all_scenes = df['scene_name'].unique()
n_frames = []
for scene in all_scenes:
    df_scene = df[df['scene_name'] == scene]
    valid = df_scene[df_scene['included_in_public_release'] == True]
    n_frames.append(len(valid))

n_frames = np.array(n_frames)
print(np.max(n_frames))
pos = np.argwhere(n_frames == np.max(n_frames))[0][0]

print('smallest scene %s' % (all_scenes[pos]))
import pdb
pdb.set_trace()
