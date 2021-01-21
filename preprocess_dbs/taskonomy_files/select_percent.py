import os
import random
import tarfile
from copy import copy

random.seed(10)
extract_percent = 0.5
ds_path = "/data/multi-domain-graph-3/datasets/Taskonomy/tiny-train/"

all_rooms = [
    "hanson", "merom", "klickitat", "onaga", "leonardo", "marstons",
    "newfields", "pinesdale", "lakeville", "cosmos", "benevolence", "pomaria",
    "tolstoy", "shelbyville", "allensville", "wainscott", "beechwood",
    "coffeen", "stockman", "hiteman", "woodbine", "lindenwood", "forkland",
    "mifflinburg", "ranchester"
]

os.makedirs("%s/sample_lists_%f" % (ds_path, extract_percent), exist_ok=True)

for room_id in all_rooms:
    tar_name = "%s_rgb.tar" % room_id
    tar_obj = tarfile.open(ds_path + tar_name)
    all_files = tar_obj.getmembers()
    absolute_number_to_extract = int(extract_percent * len(all_files))
    sampled_list = random.sample(all_files, absolute_number_to_extract)
    # sampled_list = all_files[:5]
    domain_list = {}

    domains = ["edge_texture", "depth_zbuffer", "normal", "rgb"]
    tars_files = {}
    tars_names = {}
    tars_members = {}

    for domain_id in domains:
        domain_list[domain_id] = []

        tar_name = "%s_%s.tar" % (room_id, domain_id)
        tars_files[domain_id] = tarfile.open(ds_path + tar_name)
        tars_files[domain_id].errorlevel = 2
        tars_names[domain_id] = tars_files[domain_id].getnames()

    for selected_sample in sampled_list:
        next_sample = False
        buffer_list = []
        subpaths = selected_sample.path.split("/")
        if len(subpaths) <= 1:
            continue
        filename = subpaths[1].replace("_rgb.png", "")
        for domain_id in domains:
            filename_for_domain = "%s/%s_%s.png" % (domain_id, filename,
                                                    domain_id)

            if filename_for_domain not in tars_names[domain_id]:
                next_sample = True
                print("Rejected sample", filename, domain_id)
                break

            buffer_list.append(filename_for_domain)

        if next_sample:
            continue

        for idx_buffer, domain_id in enumerate(domains):
            domain_list[domain_id].append(buffer_list[idx_buffer])
        # print("\t> Sample extracted from all domains", filename)

    for domain_id in domains:
        save_list_path = "%s/sample_lists_%f/%s_%s.txt" % (
            ds_path, extract_percent, room_id, domain_id)
        save_list_path_bash = "%s/$1/%s_%s.txt" % (ds_path, room_id, domain_id)
        with open(save_list_path, 'w') as fd:
            for listitem in domain_list[domain_id]:
                fd.write('%s\n' % listitem)

            # cmd = f'tar -xf {ds_path}/{room_id}_{domain_id}.tar  --files-from {save_list_path_bash} --transform s/point_/${room_id}_point_/ -C {ds_path}'
            # print(cmd)

    print("Done", room_id, ": ", len(domain_list["rgb"]), "files added.")
    # break
