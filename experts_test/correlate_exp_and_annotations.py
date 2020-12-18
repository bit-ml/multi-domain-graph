import os 
import numpy as np 

experts_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master-experts'
gt_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master-preproc'

tasks = os.listdir(gt_path)
tasks.sort()

for task_name in tasks:
    alt_task_name = task_name
    alt_task_name = alt_task_name.replace('edge_texture', 'edges_dexined')
    alt_task_name = alt_task_name.replace('normal', 'normals_xtc')
    alt_task_name = alt_task_name.replace('depth_zbuffer', 'depth_sgdepth')

    exp_task_path = os.path.join(experts_path, alt_task_name)
    gt_task_path = os.path.join(gt_path, task_name)

    exp_filenames = os.listdir(exp_task_path)
    exp_filenames.sort()
    gt_filenames = os.listdir(gt_task_path)
    gt_filenames.sort()
    exp_min_values = []
    exp_max_values = []
    gt_min_values = []
    gt_max_values = []
    for idx in range(len(exp_filenames)):
        exp_res = np.load(os.path.join(exp_task_path, exp_filenames[idx]))
        gt_res = np.load(os.path.join(gt_task_path, gt_filenames[idx]))
        exp_min_values.append(np.min(exp_res))
        exp_max_values.append(np.max(exp_res))
        gt_min_values.append(np.min(gt_res))
        gt_max_values.append(np.max(gt_res))
        #print("%s gt: %.2f %.2f exp: %.2f %.2f"%(alt_task_name,np.min(gt_res), np.max(gt_res), np.min(exp_res), np.max(exp_res)))

    exp_min_values = np.array(exp_min_values)
    exp_max_values = np.array(exp_max_values)
    gt_min_values = np.array(gt_min_values)
    gt_max_values = np.array(gt_max_values)

    print("%s gt: %.2f %.2f exp: %.2f %.2f"%(alt_task_name,np.mean(gt_min_values), np.mean(gt_max_values), np.mean(exp_min_values), np.mean(exp_max_values)))