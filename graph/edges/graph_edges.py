import glob
import logging
import os
import pathlib
from datetime import datetime

import numpy as np
import torch
from graph.edges.dataset2d import Domain2DDataset
from graph.edges.unet.unet_model import UNetGood, UNetMedium, UNetSmall
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
# from torchvision import transforms
from tqdm import tqdm
from utils.utils import DummySummaryWriter

RGBS_PATH = "/data/tracking-vot/"
EXPERTS_OUTPUT_PATH = "/data/experts/"

DATASET_PATH = "GOT-10k/"
TRAIN_PATH = "%s/train/" % (DATASET_PATH)
VALID_PATH = "%s/val/" % (DATASET_PATH)

no_generated = 1000
EPSILON = 1e-8


def img_for_plot(img):
    '''
    img shape NCHW, ex: torch.Size([3, 1, 256, 256])
    '''
    n, c, _, _ = img.shape
    img_view = img.view(n, c, -1)
    min_img = img_view.min(axis=2)[0][:, :, None, None]
    max_img = img_view.max(axis=2)[0][:, :, None, None]

    return (img - min_img) / (max_img - min_img)
    # return transforms.Normalize(mean=mean, std=std)(img)


# def input_normaliz(img):
#     '''
#     img shape NCHW, ex: torch.Size([3, 1, 256, 256])
#     '''
#     n, c, _, _ = img.shape
#     img_view = img.view(n, c, -1)
#     mean = img_view.mean(axis=2)[:, :, None, None]
#     std = img_view.std(axis=2)[:, :, None, None]
#     return (img - mean) / (std + EPSILON)


def generate_experts_output(experts):
    pattern = "%s/*/*00001.jpg"  # train/val:
    # pattern = "%s/*/*333.jpg"  # train/val:
    # pattern = "%s/*/*0050.jpg"  # train/val:
    # pattern = "%s/*/*0111.jpg"  # train/val:
    train_dir = "%s/%s" % (RGBS_PATH, TRAIN_PATH)
    valid_dir = "%s/%s" % (RGBS_PATH, VALID_PATH)
    for rgbs_dir_path in [train_dir, valid_dir]:
        count = 0
        rgb_paths = sorted(glob.glob(pattern % rgbs_dir_path))
        for rgb_path in tqdm(rgb_paths):
            img = Image.open(rgb_path)
            for expert in experts:
                fname = rgb_path.replace(
                    "/data/tracking-vot/",
                    "%s/%s/" % (EXPERTS_OUTPUT_PATH, expert.str_id)).replace(
                        ".jpg", ".npy")
                save_folder = os.path.dirname(fname)
                if not os.path.exists(save_folder):
                    pathlib.Path(save_folder).mkdir(parents=True,
                                                    exist_ok=True)
                if not os.path.exists(fname):
                    # e_out shape: expert.n_maps x 256 x 256
                    e_out = expert.apply_expert_one_frame(img)
                    np.save(fname, e_out)

                count += 1
            if count > no_generated * len(experts):
                break
        print("Done:", rgbs_dir_path, "pattern:", pattern, "no files",
              "rgb_paths:", len(rgb_paths))


def generate_experts_output_with_time(experts):
    ends_with = "0001.jpg"  # train/val:
    # ends_with = "1333.jpg"  # train/val:
    # ends_with = "0333.jpg"  # train/val:
    # ends_with = "0050.jpg"  # train/val:
    # ends_with = "0111.jpg"  # train/val:
    train_dir = "%s/%s" % (RGBS_PATH, TRAIN_PATH)
    valid_dir = "%s/%s" % (RGBS_PATH, VALID_PATH)
    for rgbs_dir_path in [train_dir, valid_dir]:
        all_dirs = sorted(os.listdir(rgbs_dir_path))
        count = 0

        for crt_video_dir in tqdm(all_dirs):
            if len(
                    glob.glob("%s/%s/*%s" %
                              (rgbs_dir_path, crt_video_dir, ends_with))) == 0:
                continue

            # skip already generated outputs:
            already_gen = True
            for expert in experts:
                pattern = "%s/%s/*%s" % (rgbs_dir_path, crt_video_dir,
                                         ends_with)
                fpaths = glob.glob(pattern)
                for path in fpaths:
                    path = path.replace(
                        "/data/tracking-vot/",
                        "%s/%s/" % (EXPERTS_OUTPUT_PATH, expert.str_id))

                    npy_path = path.replace(".jpg", ".npy")
                    if not os.path.exists(npy_path):
                        already_gen = False
            if already_gen:
                count += 1
                continue

            all_imgs = sorted(
                glob.glob("%s/%s/*.jpg" % (rgbs_dir_path, crt_video_dir)))
            rgb_frames = []
            for rgb_path in all_imgs:
                img = Image.open(rgb_path)
                rgb_frames.append(img)

                # needs clear pattern, without *
                if rgb_path.endswith(ends_with):
                    break

            for expert in experts:
                fname = rgb_path.replace(
                    "/data/tracking-vot/",
                    "%s/%s/" % (EXPERTS_OUTPUT_PATH, expert.str_id)).replace(
                        ".jpg", ".npy")
                save_folder = os.path.dirname(fname)
                if not os.path.exists(save_folder):
                    pathlib.Path(save_folder).mkdir(parents=True,
                                                    exist_ok=True)
                if not os.path.exists(fname):
                    # doar de tracking?!! (to change)
                    with open(
                            "%s/%s/groundtruth.txt" %
                        (rgbs_dir_path, crt_video_dir), "r") as fd:
                        start_bbox_str = fd.readline()
                        start_bbox = [
                            float(x) for x in start_bbox_str.replace(
                                "\n", "").split(",")
                        ]

                    # e_out shape: expert.n_maps x 256 x 256
                    e_out = expert.apply_expert_for_last_map(
                        rgb_frames, start_bbox)

                    np.save(fname, e_out)
                count += 1

            if count > no_generated * len(experts):
                break
        print("Done:", rgbs_dir_path, "ends_with:", ends_with)


class Edge:
    def __init__(self, expert1, expert2, device, silent):
        super(Edge, self).__init__()
        self.init_edge(expert1, expert2, device)
        self.init_loaders(bs=40, n_workers=4)

        self.lr = 5e-4
        self.optimizer = AdamW(self.net.parameters(),
                               lr=self.lr,
                               weight_decay=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           patience=10,
                                           factor=0.5,
                                           threshold=0.005,
                                           min_lr=5e-5)
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.l2_detailed_eval = nn.MSELoss(reduction='none')
        self.l1_detailed_eval = nn.L1Loss(reduction='none')

        self.global_step = 0
        self.ill_posed = False

        if silent:
            self.writer = DummySummaryWriter()
        else:
            self.writer = SummaryWriter(
                log_dir=f'runs/with_of_fwd_raft_%s_%s_%s' %
                (expert1.str_id, expert2.str_id, datetime.now()),
                flush_secs=30)

        trainable_params = sum(p.numel() for p in self.net.parameters()
                               if p.requires_grad)
        total_params = sum(p.numel() for p in self.net.parameters())

        self.writer.add_text("trainable_params", str(trainable_params))
        self.writer.add_text("total_params", str(total_params))
        self.writer.add_text("net class", str(self.net.__class__))

    def init_edge(self, expert1, expert2, device):
        self.expert1 = expert1
        self.expert2 = expert2
        self.name = "%s -> %s" % (expert1.domain_name, expert2.domain_name)
        self.net = UNetGood(n_channels=self.expert1.n_maps,
                            n_classes=self.expert2.n_maps,
                            bilinear=True).to(device)

    def init_loaders(self, bs, n_workers):
        experts = [self.expert1, self.expert2]
        train_ds = Domain2DDataset(RGBS_PATH, EXPERTS_OUTPUT_PATH, TRAIN_PATH,
                                   experts)
        valid_ds = Domain2DDataset(RGBS_PATH, EXPERTS_OUTPUT_PATH, VALID_PATH,
                                   experts)

        self.train_loader = DataLoader(train_ds,
                                       batch_size=bs,
                                       shuffle=True,
                                       num_workers=n_workers)
        self.valid_loader = DataLoader(valid_ds,
                                       batch_size=bs,
                                       shuffle=False,
                                       num_workers=n_workers)

    def train_step(self, device):
        self.net.train()

        train_l1_loss = 0
        train_l2_loss = 0

        for batch in self.train_loader:
            domain1, domain2_gt = batch
            assert domain1.shape[1] == self.net.n_channels
            assert domain2_gt.shape[1] == self.net.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            domain2_pred = self.net(domain1)
            l2_loss = self.l2(domain2_pred, domain2_gt)
            train_l2_loss += l2_loss.item()

            train_l1_loss += self.l1(domain2_pred, domain2_gt).item()
            # print("-----train_loss", train_loss, domain2_pred.shape)

            # Optimizer
            self.optimizer.zero_grad()
            l2_loss.backward()
            # nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
            self.optimizer.step()

        tag = 'Train/%s---%s' % (self.expert1.str_id, self.expert2.str_id)
        if domain1.shape[1] == 2:
            domain1 = domain1[:, 0:1]
        self.writer.add_images('%s/Input' % (tag), img_for_plot(domain1[:3]),
                               self.global_step)

        if domain2_gt.shape[1] == 2:
            domain2_gt = domain2_gt[:, 0:1]
        self.writer.add_images('%s/GT' % (tag), img_for_plot(domain2_gt[:3]),
                               self.global_step)

        if domain2_pred.shape[1] == 2:
            domain2_pred = domain2_pred[:, 0:1]
        self.writer.add_images('%s/Output' % (tag),
                               img_for_plot(domain2_pred[:3]),
                               self.global_step)
        return train_l2_loss / len(
            self.train_loader) * 100, train_l1_loss / len(
                self.train_loader) * 100

    def __str__(self):
        return 'From: %s To: %s' % (self.expert1.domain_name,
                                    self.expert2.domain_name)

    def eval_detailed(self, device):
        self.net.eval()
        eval_l2_loss = []
        eval_l1_loss = []

        for batch in self.valid_loader:
            domain1, domain2_gt = batch
            assert domain1.shape[1] == self.net.n_channels
            assert domain2_gt.shape[1] == self.net.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                domain2_pred = self.net(domain1)
            loss_l2 = self.l2_detailed_eval(domain2_pred, domain2_gt)
            loss_l1 = self.l1_detailed_eval(domain2_pred, domain2_gt)

            eval_l2_loss += loss_l2.view(domain2_pred.shape[0],
                                         -1).mean(axis=1).data.cpu()
            eval_l1_loss += loss_l1.view(domain2_pred.shape[0],
                                         -1).mean(axis=1).data.cpu()

        return eval_l2_loss * 100, eval_l1_loss * 100

    def eval_step(self, device):
        self.net.eval()
        eval_l2_loss = 0
        eval_l1_loss = 0

        for batch in self.valid_loader:
            domain1, domain2_gt = batch
            assert domain1.shape[1] == self.net.n_channels
            assert domain2_gt.shape[1] == self.net.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                domain2_pred = self.net(domain1)
            l2_loss = self.l2(domain2_pred, domain2_gt)
            l1_loss = self.l1(domain2_pred, domain2_gt)

            eval_l2_loss += l2_loss.item()
            eval_l1_loss += l1_loss.item()

        tag = 'Valid/%s---%s' % (self.expert1.str_id, self.expert2.str_id)

        if domain1.shape[1] == 2:
            domain1 = domain1[:, 0:1]
        self.writer.add_images('%s/Input' % tag, img_for_plot(domain1[:3]),
                               self.global_step)
        if domain2_gt.shape[1] == 2:
            domain2_gt = domain2_gt[:, 0:1]

        self.writer.add_images('%s/GT' % tag, img_for_plot(domain2_gt[:3]),
                               self.global_step)

        if domain2_pred.shape[1] == 2:
            domain2_pred = domain2_pred[:, 0:1]

        self.writer.add_images('%s/Output' % tag,
                               img_for_plot(domain2_pred[:3]),
                               self.global_step)

        return eval_l2_loss / len(self.valid_loader) * 100, eval_l1_loss / len(
            self.valid_loader) * 100

    def train(self, epochs, device):
        for epoch in range(epochs):
            train_l2_loss, train_l1_loss = self.train_step(device)
            self.writer.add_scalar('Train/L2_Loss', train_l2_loss,
                                   self.global_step)
            self.writer.add_scalar('Train/L1_Loss', train_l1_loss,
                                   self.global_step)

            val_l2_loss, val_l1_loss = self.eval_step(device)
            self.writer.add_scalar('Valid/L2_Loss', val_l2_loss,
                                   self.global_step)
            self.writer.add_scalar('Valid/L1_Loss', val_l1_loss,
                                   self.global_step)

            # Scheduler
            self.scheduler.step(val_l2_loss)
            self.writer.add_scalar('Train/LR',
                                   self.optimizer.param_groups[0]['lr'],
                                   self.global_step)

            # Histograms
            # if global_step % (n_train // (10 * batch_size)) == 0:
            #     for tag, value in net.named_parameters():
            #         tag = tag.replace('.', '/')
            #         writer.add_histogram('weights/' + tag,
            #                              value.data.cpu().numpy(),
            #                              global_step)
            #         writer.add_histogram('grads/' + tag,
            #                              value.grad.data.cpu().numpy(),
            #                              global_step)

            self.global_step += 1

        # if save_cp:
        #     try:
        #         os.mkdir(dir_checkpoint)
        #         logging.info('Created checkpoint directory')
        #     except OSError:
        #         pass
        #     torch.save(net.state_dict(),
        #                dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        #     logging.info(f'Checkpoint {epoch + 1} saved !')

    # self.writer.close()
