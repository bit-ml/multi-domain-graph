import os

folder1 = "/data/multi-domain-graph/models/continuari"
folder2 = "/data/multi-domain-graph/models/trained_21ian_tiny_train0.1"
delete_if_empty_path = "/data/multi-domain-graph/models/elena"

folder1_domains = os.listdir(folder1)
for domain_folder in folder1_domains:
    print("EDGE", domain_folder)
    cmd = f'mv -vn {folder1}/{domain_folder}/* {folder2}/{domain_folder}/'
    os.system(cmd)
    # print(cmd)
    # break
os.system("find %s -type d -empty -print" % delete_if_empty_path)
# os.system("find %s -type d -empty -delete" % delete_if_empty_path)
