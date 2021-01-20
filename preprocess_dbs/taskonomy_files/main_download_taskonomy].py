import os
import shutil
import sys

import numpy as np

pattern_train = ['1', '0', '0']
pattern_val = ['0', '1', '0']
pattern_test = ['0', '0', '1']

# 1. choose split: train or val or test
chosen_split = "train"

# 2. change pattern: pattern_train or pattern_val or pattern_test
pattern = pattern_train

# 3. choose tiny, medium, full, fullplus
main_splits_file = r'/data/multi-domain-graph-3/datasets/Taskonomy_info/splits_taskonomy/train_val_test_tiny.csv'

all_domains = ["rgb", "depth_zbuffer", "normal", "edge_texture"]
for domain_name in all_domains:

    main_links_file = r'/data/multi-domain-graph-3/datasets/Taskonomy_info/taskonomy_dl_links/EPFL/by_domain/epfl_%s_taskonomy.txt' % domain_name
    save_links_to_file = r'/data/multi-domain-graph-3/datasets/Taskonomy/epfl_%s_taskonomy_%s.txt' % (
        domain_name, chosen_split)

    # Run all
    csv_file = open(main_splits_file)
    lines = [line.split() for line in csv_file]
    lines = lines[1:]
    to_download_buildings = []
    for line in lines:
        data = line[0].split(',')
        building_name = data[0]
        if data[1] == pattern[0] and data[2] == pattern[1] and data[
                3] == pattern[2]:
            to_download_buildings.append(building_name)
    to_download_buildings.sort()

    print("to_download_buildings", to_download_buildings)

    correct_links = []
    found_buildings = []
    links_file = open(main_links_file, 'r')
    all_links = links_file.readlines()
    for link in all_links:
        pos = link.find('/taskonomy/')
        pos = pos + len('/taskonomy/')
        building = link[pos:]
        pos = building.find('_')
        building = building[:pos]
        if building in to_download_buildings:
            correct_links.append(link)
            found_buildings.append(building)
    found_buildings.sort()
    print("found_buildings", found_buildings)
    print("correct_links", correct_links)

    for b in to_download_buildings:
        if not b in found_buildings:
            print(b, "Not found")

    new_file = open(save_links_to_file, 'w')
    for correct_link in correct_links:
        new_file.write(correct_link)
    new_file.close()

    print("|Split %s| Domain %s| Links saved to %s" %
          (chosen_split, domain_name, save_links_to_file))
