import os
import sys
import traceback

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import experts.depth_expert
import experts.edges_expert
import experts.grayscale_expert
import experts.halftone_expert
import experts.hsv_expert
import experts.liteflownet_of_expert
import experts.normals_expert
import experts.raft_of_expert
import experts.rgb_expert
import experts.saliency_seg_expert
import experts.semantic_segmentation_expert
import experts.vmos_stm_expert

WORKING_H = 256
WORKING_W = 256

# dataset domain names
VALID_ORIG_GT_DOMAINS = [
    'rgb', 'depth_zbuffer', 'edge_texture', 'normal', 'halftone_gray',
    'segment_semantic', 'grayscale', 'hsv'
]

# our internal domain names
VALID_GT_DOMAINS = [\
    'rgb',
    'depth',
    'edges',
    'normals',
    'halftone_gray',
    'sem_seg',
    'grayscale',
    'hsv'\
]

VALID_EXPERTS_NAME = [\
    'depth_xtc',
    'edges_dexined',
    'normals_xtc',
    'sem_seg_hrnet'\
]

VALID_SPLITS_NAME = [\
    "tiny-val",
    "tiny-test",
    "tiny-train-0.15-part1",
    "tiny-train-0.15-part2",
    "tiny-train-0.15-part3"\
]

RUN_TYPE = []
EXPERTS_NAME = []
ORIG_DOMAINS = []
DOMAINS = []

usage_str = 'usage: python main_taskonomy.py type split-name exp1 exp2 ...'
#    type                   - [0/1] - 0 create preprocessed gt samples
#                                   - 1 create preprocessed experts samples
#    expi                   - name of the i'th expert / domain
#                           - should be one of the VALID_EXPERTS_NAME / VALID_GT_DOMAINS
#                           - 'all' to run all available experts / domains


def check_arguments_without_delete(argv):
    global RUN_TYPE
    global EXPERTS_NAME
    global ORIG_DOMAINS
    global DOMAINS
    global MAIN_DB_PATH
    global MAIN_GT_OUT_PATH
    global MAIN_EXP_OUT_PATH

    if len(argv) < 3:
        return 0, 'incorrect usage'

    RUN_TYPE = np.int32(argv[1])
    if not (RUN_TYPE == 0 or RUN_TYPE == 1):
        return 0, 'incorrect run type: %d' % RUN_TYPE

    split_name = argv[2]
    if split_name not in VALID_SPLITS_NAME:
        status = 0
        status_code = 'Split %s is not valid' % split_name
        return status, status_code
    print('SPLIT:', split_name)

    MAIN_DB_PATH = r'/data/multi-domain-graph-2/datasets/Taskonomy/%s' % split_name
    MAIN_GT_OUT_PATH = r'/data/multi-domain-graph-2/datasets/datasets_preproc_gt/taskonomy/%s' % split_name
    MAIN_EXP_OUT_PATH = r'/data/multi-domain-graph-2/datasets/datasets_preproc_exp/taskonomy/%s' % split_name

    if RUN_TYPE == 0:
        if argv[3] == 'all':
            ORIG_DOMAINS = []
            DOMAINS = []
            for doms in zip(VALID_ORIG_GT_DOMAINS, VALID_GT_DOMAINS):
                orig_dom_name, dom_name = doms
                ORIG_DOMAINS.append(orig_dom_name)
                DOMAINS.append(dom_name)
        else:
            potential_domains = argv[3:]
            print("potential_domains", potential_domains)
            print("VALID_GT_DOMAINS", VALID_GT_DOMAINS)
            ORIG_DOMAINS = []
            DOMAINS = []
            for i in range(len(potential_domains)):
                dom_name = potential_domains[i]
                if not dom_name in VALID_GT_DOMAINS:
                    status = 0
                    status_code = 'Domain %s is not valid' % dom_name
                    return status, status_code
                orig_dom_name = VALID_ORIG_GT_DOMAINS[VALID_GT_DOMAINS.index(
                    dom_name)]

                ORIG_DOMAINS.append(orig_dom_name)
                DOMAINS.append(dom_name)
        print("ORIG_DOMAINS", ORIG_DOMAINS)
        return 1, ''
    else:
        if argv[3] == 'all':
            EXPERTS_NAME = []
            for exp_name in VALID_EXPERTS_NAME:
                EXPERTS_NAME.append(exp_name)
        else:
            potential_experts = argv[3:]
            EXPERTS_NAME = []
            for i in range(len(potential_experts)):
                exp_name = potential_experts[i]
                if not exp_name in VALID_EXPERTS_NAME:
                    status = 0
                    status_code = 'Expert %s is not valid' % exp_name
                    return status, status_code
                EXPERTS_NAME.append(exp_name)
        return 1, ''


def get_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (WORKING_W, WORKING_H), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_pil_image(img_path):
    img = Image.open(img_path)
    img = img.resize((WORKING_H, WORKING_W))
    return img


def get_expert(exp_name):
    if exp_name == 'of_fwd_raft':
        return experts.raft_of_expert.RaftModel(full_expert=True, fwd=1)
    elif exp_name == 'of_bwd_raft':
        return experts.raft_of_expert.RaftModel(full_expert=True, fwd=0)
    elif exp_name == 'of_fwd_liteflownet':
        return experts.liteflownet_of_expert.LiteFlowNetModel(full_expert=True,
                                                              fwd=1)
    elif exp_name == 'of_bwd_liteflownet':
        return experts.liteflownet_of_expert.LiteFlowNetModel(full_expert=True,
                                                              fwd=0)
    elif exp_name == 'sseg_fcn':
        return experts.sseg_fcn_expert.FCNModel(full_expert=True)
    elif exp_name == 'sseg_deeplabv3':
        return experts.sseg_deeplabv3_expert.DeepLabv3Model(full_expert=True)
    elif exp_name == 'vmos_stm':
        return experts.vmos_stm_expert.STMModel(
            'experts/vmos_stm/STM_weights.pth', 0, 21)
    elif exp_name == 'halftone_gray':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=0)
    elif exp_name == 'halftone_rgb_basic':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=1)
    elif exp_name == 'halftone_cmyk_basic':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=2)
    elif exp_name == 'halftone_rot_gray_basic':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=3)
    elif exp_name == 'depth_sgdepth':
        sys.argv = ['']
        return experts.depth_expert.DepthModel(full_expert=True)
    elif exp_name == 'depth_xtc':
        return experts.depth_expert.DepthModelXTC(full_expert=True)
    elif exp_name == 'edges_dexined':
        return experts.edges_expert.EdgesModel(full_expert=True)
    elif exp_name == 'normals_xtc':
        return experts.normals_expert.SurfaceNormalsXTC(
            dataset_name="taskonomy", full_expert=True)
    elif exp_name == 'saliency_seg_egnet':
        return experts.saliency_seg_expert.SaliencySegmModel(full_expert=True)
    elif exp_name == 'rgb':
        return experts.rgb_expert.RGBModel(full_expert=True)
    elif exp_name == 'sem_seg_hrnet':
        return experts.semantic_segmentation_expert.SSegHRNet(full_expert=True)
    elif exp_name == 'grayscale':
        return experts.grayscale_expert.Grayscale(full_expert=True)
    elif exp_name == 'hsv':
        return experts.hsv_expert.HSVExpert(full_expert=True)


def get_data_range(in_path, right_dtype):
    filenames = os.listdir(in_path)
    filenames.sort()
    min_values = []
    max_values = []
    # search data_range only in 5k files
    filenames = filenames[:5000]
    for idx, filename in enumerate(tqdm(filenames)):
        data_path = os.path.join(in_path, filename)
        try:
            data = Image.open(data_path)
            data = np.array(data)
            if not data.dtype == right_dtype:
                print("ALTA ERROARE::: path", data_path, "index", idx,
                      "process_rgb", data.dtype)
                data = np.zeros((5, 5), dtype=right_dtype)
        except:
            # traceback.print_exc()
            print("ERROARE::: path", data_path, "index", idx, "process_rgb")
            data = np.zeros((5, 5), dtype=right_dtype)

        min_values.append(data.min())
        max_values.append(data.max())
    min_value = min(min_values)
    max_value = max(max_values)

    max_values = np.array(max_values)
    max_values[max_values == 65535] = 0
    second_max_value = np.max(max_values)
    print('range: -- min: %20.10f max: %20.10f second_max: %20.10f' %
          (min_value, max_value, second_max_value))
    return min_value, max_value, second_max_value


def process_sem_seg(in_path, out_path):
    '''
    segment_semantic/
        The semantic segmentation classes are a subset of MS COCO dataset classes. The annotations are in the form of pixel-wise object labels and are distilled from [FCIS](https://arxiv.org/pdf/1611.07709.pdf), so they should be viewed as pseudo labels, as opposed to labels done individually by human annotators (a more accurate annotation set will be released in the near future). 
    
        The annotations have 18 unique labels, which include 16 object classes, a "background" class, and an "uncertain" class. "Background" means the FCIS classifiers were certain that those pixels belong to none of the 16 objects in the dictionary. "Uncertain" means the classifiers had too low confidence for those pixels to mark them as either an object or the background -- so they could belong to any class and they should be masked during learning to not contribute to the loss in a positive or negative way. 
        
        The classes "0" and "1" mark "uncertain" and "background" pixels, respectively. The rest of the classes are specified in [this file](https://github.com/StanfordVL/taskonomy/blob/master/taskbank/assets/web_assets/pseudosemantics/coco_selected_classes.txt). 
    '''
    os.makedirs(out_path, exist_ok=True)
    filenames = os.listdir(in_path)
    filenames.sort()
    max_cls = 17.

    for idx, filename in enumerate(tqdm(filenames)):
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        if os.path.exists(out_img_path):
            continue
        try:
            img_path = os.path.join(in_path, filename)
            img = get_image(img_path)
        except:
            traceback.print_exc()
            print("ERROARE::: path", img_path, "index", idx, "process_rgb")
            img = np.zeros((WORKING_W, WORKING_H, 3), dtype=np.float32)

        img = img[:, :, 0].astype('float32') / max_cls
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        np.save(out_img_path, img[None])


def process_gt_from_expert(domain_name):
    get_exp_results(MAIN_GT_OUT_PATH, experts_name=[domain_name])

    # link it in the experts
    to_unlink = os.path.join(MAIN_EXP_OUT_PATH, domain_name)
    os.system("unlink %s" % (to_unlink))

    from_path = os.path.join(MAIN_GT_OUT_PATH, domain_name)
    os.system("ln -s %s %s" % (from_path, MAIN_EXP_OUT_PATH))


def process_rgb(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    filenames = os.listdir(in_path)
    filenames.sort()

    for idx, filename in enumerate(tqdm(filenames)):
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        if os.path.exists(out_img_path):
            continue
        try:
            img_path = os.path.join(in_path, filename)
            img = get_image(img_path)
        except:
            traceback.print_exc()
            print("ERROARE::: path", img_path, "index", idx, "process_rgb")
            img = np.zeros((WORKING_W, WORKING_H, 3), dtype=np.uint8)

        img = img.astype('float32').transpose(2, 0, 1) / 255.
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        np.save(out_img_path, img)

    # link it in the experts
    to_unlink = os.path.join(MAIN_EXP_OUT_PATH, "rgb")
    os.system("unlink %s" % (to_unlink))
    os.system("ln -s %s %s" % (out_path, MAIN_EXP_OUT_PATH))


def process_depth(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    filenames = os.listdir(in_path)
    filenames.sort()
    min_value, max_value, second_max_value = get_data_range(in_path, np.int32)
    #min_value = 102
    #max_value = 65535
    #second_max_value = 7683
    idx = 0
    for idx, filename in enumerate(filenames):
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        if os.path.exists(out_img_path):
            continue

        data_path = os.path.join(in_path, filename)
        try:
            data = Image.open(data_path)
            data = np.array(data)
        except:
            traceback.print_exc()
            print("ERROARE::: path", data_path, "index", idx, "process_depth")
            data = np.zeros((WORKING_W, WORKING_H), dtype=np.float32)

        data = torch.from_numpy(data[None]).float()
        data = torch.nn.functional.interpolate(data[None],
                                               (WORKING_H, WORKING_W))[0]
        data[data == max_value] = (min_value + second_max_value) / 2
        data = (data - min_value) / second_max_value
        data = 1 - data
        data = data.numpy()
        np.save(out_img_path, data)


def process_edges(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    filenames = os.listdir(in_path)
    filenames.sort()
    idx = 0
    _, max_value, _ = get_data_range(in_path, np.int32)
    # max_value = 11355
    for idx, filename in enumerate(tqdm(filenames)):
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        if os.path.exists(out_img_path):
            continue

        data_path = os.path.join(in_path, filename)
        try:
            data = Image.open(data_path)
            data = np.array(data)
            if not data.dtype is np.dtype('int32'):
                print("ALTA ERROARE::: path", data_path, "index", idx,
                      "process_rgb")
                data = np.zeros((WORKING_W, WORKING_H), dtype=np.float32)
        except:
            # traceback.print_exc()
            print("ERROARE::: path", data_path, "index", idx, "process_edges")
            data = np.zeros((WORKING_W, WORKING_H), dtype=np.float32)

        data = torch.from_numpy(data[None]).float()
        data = torch.nn.functional.interpolate(data[None],
                                               (WORKING_H, WORKING_W))[0]
        data = data / max_value
        data = data.numpy()

        np.save(out_img_path, data)


def process_surface_normals(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    filenames = os.listdir(in_path)
    filenames.sort()
    _, max_value, _ = get_data_range(in_path, np.uint8)
    # min_value = 0
    # max_value = 255
    idx = 0
    for idx, filename in enumerate(filenames):
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        # if os.path.exists(out_img_path):
        #     continue
        data_path = os.path.join(in_path, filename)
        try:
            data = Image.open(data_path)
            data = np.array(data)
        except:
            traceback.print_exc()
            print("ERROARE::: path", data_path, "index", idx,
                  "process_surface_normals")
            data = np.zeros((WORKING_W, WORKING_H, 3), dtype=np.float32)

        data = torch.from_numpy(data).float()
        data = data.permute(2, 0, 1)
        data = torch.nn.functional.interpolate(data[None],
                                               (WORKING_H, WORKING_W))[0]
        data = data / max_value
        data = data.numpy()
        np.save(out_img_path, data)


def get_gt_domains():
    print("get_gt_domains", ORIG_DOMAINS)
    for doms in zip(ORIG_DOMAINS, DOMAINS):
        orig_dom_name, dom_name = doms

        in_path = os.path.join(MAIN_DB_PATH, orig_dom_name)
        out_path = os.path.join(MAIN_GT_OUT_PATH, dom_name)

        if orig_dom_name == 'rgb':
            process_rgb(in_path, out_path)
        elif orig_dom_name == 'depth_zbuffer':
            process_depth(in_path, out_path)
        elif orig_dom_name == 'edge_texture':
            process_edges(in_path, out_path)
        elif orig_dom_name == 'normal':
            process_surface_normals(in_path, out_path)
        elif orig_dom_name == 'segment_semantic':
            process_sem_seg(in_path, out_path)
        elif orig_dom_name in ['grayscale', 'halftone_gray', 'hsv']:
            process_gt_from_expert(orig_dom_name)


class Dataset_ImgLevel(Dataset):
    def __init__(self, rgbs_path):
        super(Dataset_ImgLevel, self).__init__()

        filenames = os.listdir(rgbs_path)
        filenames.sort()
        self.rgbs_path = []
        for filename in filenames:
            self.rgbs_path.append(os.path.join(rgbs_path, filename))

    def __getitem__(self, index):
        try:
            rgb = get_image(self.rgbs_path[index])
        except:
            traceback.print_exc()
            print("ERROARE::: path", self.rgbs_path[index], "index", index)
            rgb = np.zeros((WORKING_W, WORKING_H, 3), dtype=np.uint8)

        return rgb, index

    def __len__(self):
        return len(self.rgbs_path)


def get_exp_results(main_exp_out_path, experts_name):
    rgbs_path = os.path.join(MAIN_DB_PATH, 'rgb')
    batch_size = 150
    dataset = Dataset_ImgLevel(rgbs_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=8)
    for exp_name in experts_name:
        print('EXPERT: %20s' % exp_name)
        if exp_name in ["sem_seg_hrnet", "normals_xtc"]:
            batch_size = 80
            if exp_name in ["sem_seg_hrnet"]:
                batch_size = 40
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=8)
        expert = get_expert(exp_name)

        exp_out_path = os.path.join(main_exp_out_path, exp_name)
        os.makedirs(exp_out_path, exist_ok=True)

        for batch_idx, (frames, indexes) in enumerate(tqdm(dataloader)):
            # skip fast (eg. for missing depth 00161500.npy)
            # if indexes[-1] < 161499:
            #     continue
            # already_exists = 0
            # for check_idx in indexes:
            #     out_path = os.path.join(exp_out_path, '%08d.npy' % check_idx)
            #     if os.path.exists(out_path):
            #         already_exists += 1
            # if already_exists == len(indexes):
            #     continue

            with torch.no_grad():
                results = expert.apply_expert_batch(frames)

            for sample in zip(results, indexes):
                expert_res, sample_idx = sample

                out_path = os.path.join(exp_out_path, '%08d.npy' % sample_idx)
                # if os.path.exists(out_path):
                #     continue

                np.save(out_path, expert_res)


def redo_surface_norm():
    EPSILON = 0.00001

    in_path = os.path.join(MAIN_GT_OUT_PATH, "normals")
    out_path = os.path.join(MAIN_GT_OUT_PATH, "normals_normalized")

    os.makedirs(out_path, exist_ok=True)

    filenames = os.listdir(in_path)
    filenames.sort()

    idx = 0
    for idx, filename in enumerate(tqdm(filenames)):
        if idx == 100:
            break
        out_img_path = os.path.join(out_path, filename)
        data_path = os.path.join(in_path, filename)
        data_np = np.load(data_path)

        aux = 2 * (data_np - 0.5)

        aux[2:, :] = 1
        aux_norm = np.linalg.norm(data_np, axis=0)[None]
        aux_renormed = aux / aux_norm

        # transform it back to RGB
        normals_maps = 0.5 * aux_renormed + 0.5

        np.save(out_img_path, normals_maps)


if __name__ == "__main__":
    status, status_code = check_arguments_without_delete(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)

    if RUN_TYPE == 0:
        get_gt_domains()
    else:
        get_exp_results(MAIN_EXP_OUT_PATH, EXPERTS_NAME)
    # redo_surface_norm()
