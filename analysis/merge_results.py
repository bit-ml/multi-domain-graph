import os
import shutil
import numpy as np
import cv2

path = '/root/code/multi-domain-graph/csv_results_hypersim_test_configs'

files = os.listdir(path)
files.sort()
idx = 0
final_path = 'final_hypersim_depth_configs.csv'
final_f = open(final_path, 'w')
for file_ in files:
    f = open(os.path.join(path, file_), 'r')
    lis = [line for line in f]

    if idx == 0:
        final_f.write('%s' % lis[0])
        idx = idx + 1
    if len(lis) > 1:
        final_f.write('%s' % lis[1])
    f.close()

final_f.close()
