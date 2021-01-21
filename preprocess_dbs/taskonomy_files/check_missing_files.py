import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

ds_name = "tiny-val"

all_files_folder = "/data/multi-domain-graph-3/datasets/Taskonomy/%s/rgb/" % ds_name
missing_files_folder = "/data/multi-domain-graph-3/datasets/Taskonomy/%s/edge_texture/" % ds_name
all_tars_with_missing_files = glob.glob(
    "/data/multi-domain-graph-3/datasets/Taskonomy/%s/*_edge_texture.tar" %
    ds_name)

all_tars_with_missing_files.sort(reverse=True)

all_files = os.listdir(all_files_folder)
not_all_files = os.listdir(missing_files_folder)
for idx, fname in enumerate(all_files):
    searched_file = fname.replace("_rgb.", "_edge_texture.")
    if searched_file not in not_all_files:
        print("[Missing file!!]", searched_file)
        info_array = searched_file.split("_")
        room_name, point_id, view_id = info_array[0], info_array[
            2], info_array[4]

        other_img = None
        for view_id_try in range(10):
            other_file = searched_file.replace(
                "_view_%s_domain_" % view_id, "_view_%d_domain_" % view_id_try)
            try:
                other_img = Image.open(missing_files_folder + "/" + other_file)
            except:
                continue

            break
        other_img_np = np.zeros_like(np.array(other_img))
        np.save(missing_files_folder + "/" + searched_file, other_img_np)
        # break

# # other: check one single file
# searched_file = "point_2000_view_0_domain_edge_texture.png"

# for idx_arch, tar_arch in enumerate(all_tars_with_missing_files):
#     # print("Search in >>>>>", tar_arch, "and",
#     #       all_tars_having_all_files[idx_arch])
#     cmd = "tar -tvf %s 'edge_texture/%s'" % (tar_arch, searched_file)
#     # print(cmd)
#     ret1 = os.system(cmd)
