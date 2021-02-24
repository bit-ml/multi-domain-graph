import os
import sys
import traceback
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#import experts.cartoon_expert
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
#import experts.semantic_segmentation_expert
import experts.sobel_expert
import experts.superpixel_expert
import experts.vmos_stm_expert

depth_gt_th_50 = 1.5025
depth_exp_th_50 = 0.0999
depth_scale_factor_exp = depth_gt_th_50 / depth_exp_th_50
depth_gt_th_5 = 0.6077
depth_gt_th_95 = 3.6066
depth_exp_th_5 = 0.6889
depth_exp_th_95 = 3.6226

WORKING_H = 256
WORKING_W = 256

# dataset domain names
VALID_ORIG_GT_DOMAINS = [
    'rgb', 'depth', 'normals', "halftone_gray", "grayscale", "hsv"
]

# our internal domain names
VALID_GT_DOMAINS = [\
    'rgb',
    'depth_n',
    'normals',
    'halftone_gray',
    'grayscale',
    'hsv'\
]

VALID_EXPERTS_NAME = [\
    'depth_n_xtc',
    'depth_sgdepth',
    'edges_dexined',
    'normals_xtc',
    'sem_seg_hrnet',
    'cartoon_wb',
    'superpixel_fcn',
    'sobel_small',
    'sobel_medium',
    'sobel_large'\
]
VALID_SPLITS_NAME = ["val", "test", "train"]

RUN_TYPE = []
EXPERTS_NAME = []
ORIG_DOMAINS = []
DOMAINS = []
SPLIT_NAME = ''

usage_str = 'usage: python main_taskonomy.py type split-name exp1 exp2 ...'
#    type                   - [0/1] - 0 create preprocessed gt samples
#                                   - 1 create preprocessed experts samples
#    expi                   - name of the i'th expert / domain
#                           - should be one of the VALID_EXPERTS_NAME / VALID_GT_DOMAINS
#                           - 'all' to run all available experts / domains
#    split-name             - should be one of the VALID_SPLITS_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SURFNORM_KERNEL = torch.from_numpy(
    np.array([
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ]))[:, np.newaxis, ...].to(dtype=torch.float32, device=device)


def check_arguments_without_delete(argv):
    global RUN_TYPE
    global EXPERTS_NAME
    global ORIG_DOMAINS
    global DOMAINS
    global MAIN_DB_PATH
    global MAIN_GT_OUT_PATH
    global MAIN_EXP_OUT_PATH
    global SPLIT_NAME

    if len(argv) < 4:
        return 0, 'incorrect usage'

    RUN_TYPE = np.int32(argv[1])
    if not (RUN_TYPE == 0 or RUN_TYPE == 1 or RUN_TYPE == 2 or RUN_TYPE == 3):
        return 0, 'incorrect run type: %d' % RUN_TYPE

    split_name = argv[2]
    if split_name not in VALID_SPLITS_NAME:
        status = 0
        status_code = 'Split %s is not valid. Valid ones are: %s' % (
            split_name, VALID_SPLITS_NAME)
        return status, status_code
    SPLIT_NAME = split_name

    MAIN_DB_PATH = r'/data/multi-domain-graph-2/datasets/replica_raw/%s' % split_name
    MAIN_GT_OUT_PATH = r'/data/multi-domain-graph-2/datasets/datasets_preproc_gt/replica/%s' % split_name
    MAIN_EXP_OUT_PATH = r'/data/multi-domain-graph-2/datasets/datasets_preproc_exp/replica/%s' % split_name

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
    elif RUN_TYPE == 1:
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
    else:
        return 1, ''


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
    elif exp_name == 'sem_seg_fcn':
        return experts.semantic_segmentation_expert.FCNModel(full_expert=True)
    elif exp_name == 'sem_seg_deeplabv3':
        return experts.semantic_segmentation_expert.DeepLabv3Model(
            full_expert=True)
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
    elif exp_name == 'depth_xtc' or exp_name == 'depth_n_xtc':
        return experts.depth_expert.DepthModelXTC(full_expert=True)
    elif exp_name == 'edges_dexined':
        return experts.edges_expert.EdgesModel(full_expert=True)
    elif exp_name == 'normals_xtc':
        return experts.normals_expert.SurfaceNormalsXTC(dataset_name="replica",
                                                        full_expert=True)
    elif exp_name == 'saliency_seg_egnet':
        return experts.saliency_seg_expert.SaliencySegmModel(full_expert=True)
    elif exp_name == 'rgb':
        return experts.rgb_expert.RGBModel(full_expert=True)
    elif exp_name == 'sem_seg_hrnet':
        return experts.semantic_segmentation_expert.SSegHRNet(
            dataset_name="replica", full_expert=True)
    elif exp_name == 'grayscale':
        return experts.grayscale_expert.Grayscale(full_expert=True)
    elif exp_name == 'hsv':
        return experts.hsv_expert.HSVExpert(full_expert=True)
    elif exp_name == 'cartoon_wb':
        return experts.cartoon_expert.CartoonWB(full_expert=True)
    elif exp_name == 'sobel_small':
        return experts.sobel_expert.SobelEdgesExpertSigmaSmall()
    elif exp_name == 'sobel_medium':
        return experts.sobel_expert.SobelEdgesExpertSigmaMedium()
    elif exp_name == 'sobel_large':
        return experts.sobel_expert.SobelEdgesExpertSigmaLarge()
    elif exp_name == 'superpixel_fcn':
        return experts.superpixel_expert.SuperPixel()


def depth_to_surface_normals(depth, surfnorm_scalar=256):
    with torch.no_grad():
        surface_normals = F.conv2d(depth,
                                   surfnorm_scalar * SURFNORM_KERNEL,
                                   padding=1)
        surface_normals[:, 2, ...] = 1
        surface_normals = surface_normals / surface_normals.norm(dim=1,
                                                                 keepdim=True)
    return surface_normals


def process_gt_from_expert(domain_name):
    get_exp_results(MAIN_GT_OUT_PATH, experts_name=[domain_name])

    # link it in the experts
    to_unlink = os.path.join(MAIN_EXP_OUT_PATH, domain_name)
    os.system("unlink %s" % (to_unlink))

    from_path = os.path.join(MAIN_GT_OUT_PATH, domain_name)
    os.system("ln -s %s %s" % (from_path, MAIN_EXP_OUT_PATH))


def process_rgb(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    os.system("cp -r '%s/'* '%s'" % (in_path, out_path))

    # link it in the experts
    to_unlink = os.path.join(MAIN_EXP_OUT_PATH, "rgb")
    os.system("unlink %s" % (to_unlink))
    os.system("ln -s %s %s" % (out_path, MAIN_EXP_OUT_PATH))


class GT_DepthDataset(Dataset):
    def __init__(self, depth_path, split_name):
        super(GT_DepthDataset, self).__init__()
        self.th_5 = depth_gt_th_5
        self.th_95 = depth_gt_th_95
        if split_name == 'valid':
            split_name = 'val'
        glob_pattern = '%s/*.npy' % (depth_path)
        self.depth_paths = sorted(glob.glob(glob_pattern))

    def __getitem__(self, index):
        depth = np.load(self.depth_paths[index])
        depth[depth == 0] = float("nan")
        depth = depth - self.th_5
        depth = depth / (self.th_95 - self.th_5)
        # depth = depth / 15.625 - old version
        return depth

    def __len__(self):
        return len(self.depth_paths)


def process_depth(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)

    depth_dataset = GT_DepthDataset(in_path, SPLIT_NAME)
    dataloader = DataLoader(depth_dataset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=20,
                            drop_last=False)
    files = os.listdir(in_path)
    files.sort()

    file_idx = 0
    for batch in tqdm(dataloader):
        depth = batch
        for i in range(depth.shape[0]):
            depth_ = depth[i]
            depth_ = np.array(depth_)
            np.save(os.path.join(out_path, '%08d.npy' % file_idx), depth_)
            file_idx += 1


def process_surface_normals(main_db_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    depth_full_path = os.path.join(main_db_path, "depth")

    batch_size = 500
    dataset = DatasetDepth(depth_full_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=8)
    with torch.no_grad():
        for batch_idx, (depth_frames, indexes) in enumerate(tqdm(dataloader)):
            # adjust depth
            depth_frames = 1 - depth_frames / 14.
            depth_frames = depth_frames.to(device)

            normals_frames = depth_to_surface_normals(depth_frames[:, None])
            normals_frames = 0.5 * normals_frames + 0.5

            # permute it to match the normals expert
            permutation = [1, 0, 2]
            normals_imgs = normals_frames.data.cpu().numpy()[:, permutation]

            # SAVE Normals npy
            for sample in zip(normals_imgs, indexes):
                normals_img, sample_idx = sample

                normals_img_path = os.path.join(out_path,
                                                '%08d.npy' % sample_idx)
                # TODO: save all batch, in smtg with workers like get_item
                np.save(normals_img_path, normals_img)


def get_gt_domains():
    print("get_gt_domains", ORIG_DOMAINS)
    for doms in zip(ORIG_DOMAINS, DOMAINS):
        orig_dom_name, dom_name = doms

        in_path = os.path.join(MAIN_DB_PATH, orig_dom_name)
        out_path = os.path.join(MAIN_GT_OUT_PATH, dom_name)

        if orig_dom_name == 'rgb':
            process_rgb(in_path, out_path)
        elif orig_dom_name == 'depth':
            process_depth(in_path, out_path)
        elif orig_dom_name == 'normals':
            process_surface_normals(MAIN_DB_PATH, out_path)
        elif orig_dom_name in ['grayscale', 'halftone_gray', 'hsv']:
            process_gt_from_expert(orig_dom_name)


class DatasetDepth(Dataset):
    def __init__(self, depth_paths):
        super(DatasetDepth, self).__init__()

        filenames = os.listdir(depth_paths)
        filenames.sort()
        self.depth_paths = []
        for filename in filenames:
            self.depth_paths.append(os.path.join(depth_paths, filename))

    def __getitem__(self, index):
        depth_npy = np.load(self.depth_paths[index])
        index_of_file = int(self.depth_paths[index].split("/")[-1].replace(
            ".npy", ""))
        return depth_npy, index_of_file

    def __len__(self):
        return len(self.depth_paths)


class Dataset_ImgLevel(Dataset):
    def __init__(self, rgbs_path):
        super(Dataset_ImgLevel, self).__init__()

        filenames = os.listdir(rgbs_path)
        filenames.sort()
        self.rgbs_path = []
        for filename in filenames:
            self.rgbs_path.append(os.path.join(rgbs_path, filename))

    def __getitem__(self, index):
        rgb = np.load(self.rgbs_path[index])
        index_of_file = int(self.rgbs_path[index].split("/")[-1].replace(
            ".npy", ""))
        return rgb, index_of_file

    def __len__(self):
        return len(self.rgbs_path)


def post_process_depth_xtc_fct(data):
    data = data * depth_scale_factor_exp
    data = data - depth_exp_th_5
    data = data / (depth_exp_th_95 - depth_exp_th_5)
    return data


def get_exp_results(main_exp_out_path, experts_name):
    with torch.no_grad():
        rgbs_path = os.path.join(MAIN_DB_PATH, 'rgb')
        batch_size = 150

        dataset = Dataset_ImgLevel(rgbs_path)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=8)

        for exp_name in experts_name:
            if exp_name in ["sem_seg_hrnet", "normals_xtc"]:
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=80,
                                                         shuffle=False,
                                                         drop_last=False,
                                                         num_workers=8)
            print('EXPERT: %20s' % exp_name)
            expert = get_expert(exp_name)

            if exp_name == 'depth_n_xtc':
                post_process_fct = post_process_depth_xtc_fct
            else:
                post_process_fct = lambda x: x
            exp_out_path = os.path.join(main_exp_out_path, exp_name)
            os.makedirs(exp_out_path, exist_ok=True)

            for batch_idx, (frames, indexes) in enumerate(tqdm(dataloader)):
                # skip fast (eg. for missing depth 00161500.npy)
                # if indexes[-1] < 161499:
                #     continue

                frames = frames.permute(0, 2, 3, 1) * 255.
                results = expert.apply_expert_batch(frames)
                results = post_process_fct(results)
                for sample in zip(results, indexes):
                    expert_res, sample_idx = sample

                    out_path = os.path.join(exp_out_path,
                                            '%08d.npy' % sample_idx)

                    np.save(out_path, expert_res)


if __name__ == "__main__":
    status, status_code = check_arguments_without_delete(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)
    print(MAIN_DB_PATH)

    if RUN_TYPE == 0:
        get_gt_domains()
    elif RUN_TYPE == 1:
        get_exp_results(MAIN_EXP_OUT_PATH, EXPERTS_NAME)
