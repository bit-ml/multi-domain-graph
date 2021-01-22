import sys
import shutil
import os
import numpy as np

path = r'/data/multi-domain-graph-3/datasets/datasets_preproc_exp/taskonomy/tiny-test-ok/rgb'

files = os.listdir(path)
files.sort()
for file_ in files:
    print(file_)
    v = np.load(os.path.join(path, file_))
    n_chs, h, w = v.shape
    if (not n_chs == 3) or (not h == 256) or (not w == 256):
        print(file_)
