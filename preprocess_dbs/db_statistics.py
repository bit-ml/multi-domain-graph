import os
import shutil
import sys
import numpy as np

db_main_path = r'/data/tracking-vot/GOT-10k/train'


def get_db_statistics(db_main_path):
    videos = os.listdir(db_main_path)
    videos.sort()
    print('# videos: %d' % (len(videos)))

    n_frames_per_vid = []
    for vid_name in videos:
        if vid_name == 'list.txt':
            continue
        filenames = os.listdir(os.path.join(db_main_path, vid_name))
        n_frames_per_vid.append(len(filenames))

    n_frames_per_vid = np.array(n_frames_per_vid)

    print('Avg #frames per video: %d' % (np.mean(n_frames_per_vid)))
    print('Min #frames per video: %d' % (np.min(n_frames_per_vid)))
    print('Max #frames per video: %d' % (np.max(n_frames_per_vid)))

    print('#frames in DB: %d' % (np.sum(n_frames_per_vid)))


exp_main_path = r'/data/multi-domain-graph/datasets/datasets_preproc_exp/GOT-10k/val'


def check_file_sizes(exp_main_path):
    experts = os.listdir(exp_main_path)
    experts.sort()

    for exp_name in experts:
        exp_path = os.path.join(exp_main_path, exp_name)
        sizes = []
        videos = os.listdir(exp_path)
        videos.sort()
        for vid_name in videos:
            vid_path = os.path.join(exp_path, vid_name)
            filenames = os.listdir(vid_path)
            filenames.sort()
            for filename in filenames:
                sizes.append(os.path.getsize(os.path.join(vid_path, filename)))

        sizes = np.array(sizes)

        print('Expert: %20s - avg size %20d min size %20d max size %20d' %
              (exp_name, np.mean(sizes), np.min(sizes), np.max(sizes)))


if __name__ == "__main__":
    #get_db_statistics()
    check_file_sizes(exp_main_path)
