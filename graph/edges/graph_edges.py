import glob
import logging
import os
import pathlib
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
from graph.edges.dataset2d import (Domain2DDataset, DomainTestDataset,
                                   DomainTrainNextIterDataset)
from graph.edges.unet.unet_model import UNetGood
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import utils
from utils.utils import EnsembleFilter_TwdExpert_SSIM_Mixed, img_for_plot
from utils.utils import EnsembleFilter_TwdExpert_SSIM_Mixed_Normalized
from utils.utils import EnsembleFilter_TwdExpert_SSIM_Mixed_Normalized_Th
from utils.utils import EnsembleFilter_TwdExpert_L1, EnsembleFilter_Equal
from utils.utils import EnsembleFilter_TwdExpert_SSIM_Normalized_Th
from utils.utils import EnsembleFilter_TwdExpert_MSSIM_Mixed_Normalized


class Edge:
    def __init__(self, config, expert1, expert2, device, rnd_sampler, silent,
                 valid_shuffle, iter_no, bs_test):
        super(Edge, self).__init__()
        self.config = config
        self.silent = silent

        ensemble_fct = config.get('Ensemble', 'ensemble_fct')
        self.ensemble_filter = None
        if ensemble_fct == 'ssim_maps_twd_exp_mixed_nn_normalized':
            self.ensemble_filter = EnsembleFilter_TwdExpert_SSIM_Mixed_Normalized(
                0.5)
        elif ensemble_fct == 'ssim_maps_twd_exp_mixed_nn_normalized_th':
            self.ensemble_filter = EnsembleFilter_TwdExpert_SSIM_Mixed_Normalized_Th(
                0.5)
        elif ensemble_fct == 'l1_maps_twd_exp_mixed_nn_normalized_th':
            self.ensemble_filter = EnsembleFilter_TwdExpert_L1(0.5)
        elif ensemble_fct == 'equal_maps_mixed_nn_normalized_th':
            self.ensemble_filter = EnsembleFilter_Equal(0.5)
        elif ensemble_fct == 'ssim_maps_twd_exp_mixed_nn':
            self.ensemble_filter = EnsembleFilter_TwdExpert_SSIM_Mixed(0.5)
        elif ensemble_fct == 'ssim_maps_twd_exp_nn_normalized_th':
            self.ensemble_filter = EnsembleFilter_TwdExpert_SSIM_Normalized_Th(
                0.5)
        elif ensemble_fct == 'mssim_maps_twd_exp_mixed_nn_normalized':
            self.ensemble_filter = EnsembleFilter_TwdExpert_MSSIM_Mixed_Normalized(
                0.5)
        if not self.ensemble_filter == None:
            self.ensemble_filter = nn.DataParallel(self.ensemble_filter)

        self.init_edge(expert1, expert2, device)
        self.init_loaders(bs=100 * torch.cuda.device_count(),
                          bs_test=bs_test * torch.cuda.device_count(),
                          n_workers=10,
                          rnd_sampler=rnd_sampler,
                          valid_shuffle=valid_shuffle,
                          iter_no=iter_no)

        learning_rate = config.getfloat('Training', 'learning_rate')
        optimizer_type = config.get('Training', 'optimizer')
        weight_decay = config.getfloat('Training', 'weight_decay')
        reduce_lr_patience = config.getfloat('Training', 'reduce_lr_patience')
        reduce_lr_factor = config.getfloat('Training', 'reduce_lr_factor')
        reduce_lr_threshold = config.getfloat('Training',
                                              'reduce_lr_threshold')
        reduce_lr_min_lr = config.getfloat('Training', 'reduce_lr_min_lr')
        momentum = config.getfloat('Training', 'momentum')
        amsgrad = config.getboolean('Training', 'amsgrad')

        self.lr = learning_rate
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(),
                                       lr=self.lr,
                                       weight_decay=weight_decay,
                                       nesterov=True,
                                       momentum=momentum)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(),
                                        lr=self.lr,
                                        weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            if amsgrad:
                self.optimizer = optim.AdamW(self.net.parameters(),
                                             lr=self.lr,
                                             weight_decay=weight_decay,
                                             amsgrad=True)
            else:
                self.optimizer = optim.AdamW(self.net.parameters(),
                                             lr=self.lr,
                                             weight_decay=weight_decay)

        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           patience=reduce_lr_patience,
                                           factor=reduce_lr_factor,
                                           threshold=reduce_lr_threshold,
                                           min_lr=reduce_lr_min_lr)

        # print("optimizer", self.optimizer)
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.l2_detailed_eval = nn.MSELoss(reduction='none')
        self.l1_detailed_eval = nn.L1Loss(reduction='none')

        self.global_step = 0
        self.ill_posed = False
        self.in_edge_weights = []
        self.in_edge_src_identifiers = []

        self.trained = False

        self.load_model_dir = os.path.join(
            config.get('Edge Models', 'load_path'),
            '%s_%s' % (expert1.identifier, expert2.identifier))

        src_domain_restr = config.get('Training', 'src_domain_restr')

        if config.getboolean('Edge Models', 'save_models'):
            self.save_model_dir = os.path.join(
                config.get('Edge Models', 'save_path'),
                config.get('Run id', 'datetime'),
                '%s_%s' % (expert1.identifier, expert2.identifier))

            if not os.path.exists(self.save_model_dir):
                if (config.getboolean('Training', 'restr_src_domain')
                        and self.expert1.domain_name == src_domain_restr
                    ) or not config.getboolean('Training', 'restr_src_domain'):
                    os.makedirs(self.save_model_dir)

            self.save_epochs_distance = config.getint('Edge Models',
                                                      'save_epochs_distance')

        # trainable_params = sum(p.numel() for p in self.net.parameters()
        #                        if p.requires_grad)
        # total_params = sum(p.numel() for p in self.net.parameters())

    def init_edge(self, expert1, expert2, device):
        self.expert1 = expert1
        self.expert2 = expert2
        self.name = "%s -> %s" % (expert1.identifier, expert2.identifier)
        self.net = UNetGood(n_channels=self.expert1.n_maps,
                            n_classes=self.expert2.n_maps,
                            bilinear=True).to(device)
        self.net = nn.DataParallel(self.net)
        total_params = sum(p.numel() for p in self.net.parameters()) / 1e+6
        trainable_params = sum(p.numel() for p in self.net.parameters()) / 1e+6
        print("\tNumber of parameters %.2fM (Trainable %.2fM)" %
              (total_params, trainable_params))

    def copy_model(self, device):
        self.prev_net = UNetGood(n_channels=self.expert1.n_maps,
                                 n_classes=self.expert2.n_maps,
                                 bilinear=True).to(device)
        self.prev_net = nn.DataParallel(self.prev_net)
        self.prev_net.load_state_dict(self.net.state_dict())

    def init_loaders(self, bs, bs_test, n_workers, rnd_sampler, valid_shuffle,
                     iter_no):
        experts = [self.expert1, self.expert2]
        iter_2_src_data = self.config.getint('Training2Iters',
                                             'iter_2_src_data')
        if self.config.getboolean('Training2Iters',
                                  'train_2_iters') and iter_no == 2:
            if iter_2_src_data == 1:
                NEXT_ITER_SRC_TRAIN_PATH = self.config.get(
                    'Training2Iters', 'NEXT_ITER_SRC_TRAIN_PATH')
            elif iter_2_src_data == 2:
                NEXT_ITER_SRC_TRAIN_PATH = self.config.get(
                    'Training2Iters', 'NEXT_ITER_DST_TRAIN_PATH')
            NEXT_ITER_DST_TRAIN_PATH = self.config.get(
                'Training2Iters', 'NEXT_ITER_DST_TRAIN_PATH')
            NEXT_ITER_DB_PATH = self.config.get('Training2Iters',
                                                'NEXT_ITER_DB_PATH')
            FIRST_K_NEXT_ITER = self.config.getint('Training2Iters',
                                                   'FIRST_K_NEXT_ITER')
            train_ds = DomainTrainNextIterDataset(NEXT_ITER_SRC_TRAIN_PATH,
                                                  NEXT_ITER_DST_TRAIN_PATH,
                                                  NEXT_ITER_DB_PATH,
                                                  experts,
                                                  FIRST_K_NEXT_ITER,
                                                  iter_no=iter_no)
        else:
            FIRST_K_TRAIN = self.config.getint('Paths', 'FIRST_K_TRAIN')
            TRAIN_PATH = self.config.get('Paths', 'TRAIN_PATH')
            TRAIN_PATTERNS = self.config.get('Paths',
                                             'TRAIN_PATTERNS').split(",")
            train_ds = Domain2DDataset(TRAIN_PATH,
                                       experts,
                                       TRAIN_PATTERNS,
                                       FIRST_K_TRAIN,
                                       iter_no=iter_no)
        VALID_PATH = self.config.get('Paths', 'VALID_PATH')
        VALID_PATTERNS = self.config.get('Paths', 'VALID_PATTERNS').split(",")
        FIRST_K_VAL = self.config.getint('Paths', 'FIRST_K_VAL')
        valid_ds = Domain2DDataset(VALID_PATH,
                                   experts,
                                   VALID_PATTERNS,
                                   FIRST_K_VAL,
                                   iter_no=iter_no)
        print("\tTrain ds", len(train_ds), "==========")
        print("\tValid ds", len(valid_ds), "==========")

        self.train_loader = DataLoader(train_ds,
                                       batch_size=bs,
                                       shuffle=True,
                                       num_workers=n_workers)
        self.valid_loader = DataLoader(
            valid_ds,
            batch_size=bs_test,
            shuffle=valid_shuffle,
            # sampler=rnd_sampler,
            num_workers=n_workers)

        if self.config.getboolean('Training2Iters',
                                  'train_2_iters') and iter_no == 1:
            NEXT_ITER_SRC_TRAIN_PATH = self.config.get(
                'Training2Iters', 'NEXT_ITER_SRC_TRAIN_PATH')
            NEXT_ITER_DB_PATH = self.config.get('Training2Iters',
                                                'NEXT_ITER_DB_PATH')
            NEXT_ITER_TRAIN_PATTERNS = self.config.get(
                'Training2Iters', 'NEXT_ITER_TRAIN_PATTERNS')
            FIRST_K_NEXT_ITER = self.config.getint('Training2Iters',
                                                   'FIRST_K_NEXT_ITER')
            next_iter_ds = Domain2DDataset(os.path.join(
                NEXT_ITER_SRC_TRAIN_PATH, NEXT_ITER_DB_PATH),
                                           experts,
                                           NEXT_ITER_TRAIN_PATTERNS,
                                           FIRST_K_NEXT_ITER,
                                           iter_no=iter_no)
            print("\tNext iter ds", len(next_iter_ds))
            self.next_iter_loader = DataLoader(next_iter_ds,
                                               batch_size=bs_test,
                                               shuffle=False,
                                               num_workers=n_workers)
        else:
            self.next_iter_loader = None
            print("\tNext iter ds 0")
        if self.config.getboolean(
                'Training2Iters',
                'train_2_iters') and iter_no == 1 and iter_2_src_data == 2:
            EXPERTS_OUTPUT_PATH_TEST = self.config.get(
                'Paths', 'EXPERTS_OUTPUT_PATH_TEST')
            TEST_PATH = self.config.get('Paths', 'TEST_PATH')
            TEST_PATTERNS = self.config.get('Paths', 'TEST_PATTERNS')
            FIRST_K_TEST = self.config.getint('Paths', 'FIRST_K_TEST')
            test_no_gt_ds = Domain2DDataset(os.path.join(
                EXPERTS_OUTPUT_PATH_TEST, TEST_PATH),
                                            experts,
                                            TEST_PATTERNS,
                                            FIRST_K_TEST,
                                            iter_no=iter_no)
            print("\tTest no gt ds", len(test_no_gt_ds))
            self.test_no_gt_loader = DataLoader(test_no_gt_ds,
                                                batch_size=bs_test,
                                                shuffle=False,
                                                num_workers=n_workers)
        else:
            self.test_no_gt_loader = None
            print("\tTest no gt ds 0")

        PREPROC_GT_PATH_TEST = self.config.get('Paths', 'PREPROC_GT_PATH_TEST')
        if self.config.getboolean('Training2Iters',
                                  'train_2_iters') and iter_no == 2:
            EXPERTS_OUTPUT_PATH_TEST = self.config.get(
                'Training2Iters', 'ENSEMBLE_OUTPUT_PATH_TEST')
        else:
            EXPERTS_OUTPUT_PATH_TEST = self.config.get(
                'Paths', 'EXPERTS_OUTPUT_PATH_TEST')
        TEST_PATH = self.config.get('Paths', 'TEST_PATH')
        FIRST_K_TEST = self.config.getint('Paths', 'FIRST_K_TEST')
        test_ds = DomainTestDataset(PREPROC_GT_PATH_TEST,
                                    EXPERTS_OUTPUT_PATH_TEST,
                                    TEST_PATH,
                                    experts,
                                    FIRST_K_TEST,
                                    iter_no=1)
        print("\tTest ds", len(test_ds), "==========")

        if test_ds.available:
            self.test_loader = DataLoader(test_ds,
                                          batch_size=bs_test,
                                          shuffle=False,
                                          num_workers=n_workers)

        else:
            self.test_loader = None

    def save_model(self, epoch):
        if not self.config.getboolean('Edge Models', 'save_models'):
            return

        if epoch % self.save_epochs_distance == 0:
            path = os.path.join(self.save_model_dir, 'epoch_%05d.pth' % epoch)
            torch.save(self.net.state_dict(), path)
            print("Model saved at %s" % path)

    def __str__(self):
        return '[%s To: %s]' % (self.expert1.domain_name,
                                self.expert2.domain_name)

    def train_step(self, device, writer, wtag):
        self.net.train()

        train_l1_loss = 0
        train_l2_loss = 0
        for batch in self.train_loader:
            domain1, domain2_gt = batch
            assert domain1.shape[1] == self.net.module.n_channels
            assert domain2_gt.shape[1] == self.net.module.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            domain2_pred = self.net(domain1)
            l2_loss = self.l2(domain2_pred, domain2_gt)
            l1_loss = self.l1(domain2_pred, domain2_gt)

            train_l2_loss += l2_loss.item()

            # with torch.no_grad():
            train_l1_loss += l1_loss.item()
            # print("-----train_loss", train_loss, domain2_pred.shape)

            # Optimizer
            self.optimizer.zero_grad()
            all_losses = l1_loss + l2_loss
            all_losses.backward()
            self.optimizer.step()

        writer.add_images('Train_%s/Input' % (wtag), img_for_plot(domain1[:3]),
                          self.global_step)

        writer.add_images('Train_%s/GT_EXPERT' % (wtag),
                          img_for_plot(domain2_gt[:3]), self.global_step)

        writer.add_images('Train_%s/Output' % (wtag),
                          img_for_plot(domain2_pred[:3]), self.global_step)

        return train_l2_loss / len(
            self.train_loader) * 100, train_l1_loss / len(
                self.train_loader) * 100

    def eval_step(self, device, writer, wtag):
        self.net.eval()
        eval_l2_loss = 0
        eval_l1_loss = 0

        for batch in self.valid_loader:
            domain1, domain2_gt = batch
            assert domain1.shape[1] == self.net.module.n_channels
            assert domain2_gt.shape[1] == self.net.module.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                domain2_pred = self.net(domain1)
            l2_loss = self.l2(domain2_pred, domain2_gt)
            l1_loss = self.l1(domain2_pred, domain2_gt)

            eval_l2_loss += l2_loss.item()
            eval_l1_loss += l1_loss.item()

        writer.add_images('Valid_%s/Input' % wtag, img_for_plot(domain1[:3]),
                          self.global_step)

        writer.add_images('Valid_%s/GT_EXPERT' % wtag,
                          img_for_plot(domain2_gt[:3]), self.global_step)

        writer.add_images('Valid_%s/Output' % wtag,
                          img_for_plot(domain2_pred[:3]), self.global_step)

        return eval_l2_loss / len(self.valid_loader) * 100, eval_l1_loss / len(
            self.valid_loader) * 100

    def test_step(self, device, writer, wtag):
        """Currently should work as eval_step
        """
        self.net.eval()
        test_l2_loss = 0
        test_l1_loss = 0

        for batch in self.test_loader:
            domain1, domain2_gt, _ = batch
            assert domain1.shape[1] == self.net.module.n_channels
            assert domain2_gt.shape[1] == self.net.module.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                domain2_pred = self.net(domain1)
            l2_loss = self.l2(domain2_pred, domain2_gt)
            l1_loss = self.l1(domain2_pred, domain2_gt)

            test_l2_loss += l2_loss.item()
            test_l1_loss += l1_loss.item()

        writer.add_images('Test_%s/Input' % wtag, img_for_plot(domain1[:3]),
                          self.global_step)

        writer.add_images('Test_%s/GT' % wtag, img_for_plot(domain2_gt[:3]),
                          self.global_step)

        writer.add_images('Test_%s/Output' % wtag,
                          img_for_plot(domain2_pred[:3]), self.global_step)

        return test_l2_loss / len(self.test_loader) * 100, test_l1_loss / len(
            self.test_loader) * 100

    def train(self, start_epoch, n_epochs, device, writer, eval_test):
        self.global_step = start_epoch
        wtag = '%s_%s' % (self.expert1.identifier, self.expert2.identifier)
        epoch = 0
        for epoch in range(n_epochs):
            # 1. Train
            train_l2_loss, train_l1_loss = self.train_step(
                device, writer, wtag)
            writer.add_scalar("Train_%s/L2_Loss" % wtag, train_l2_loss,
                              self.global_step)
            writer.add_scalar('Train_%s/L1_Loss' % wtag, train_l1_loss,
                              self.global_step)

            # Save model
            self.save_model(start_epoch + epoch + 1)

            # 2. Evaluate - validation set - pseudo gt from experts
            val_l2_loss, val_l1_loss = self.eval_step(device, writer, wtag)
            writer.add_scalar('Valid_%s/L2_Loss' % wtag, val_l2_loss,
                              self.global_step)
            writer.add_scalar('Valid_%s/L1_Loss' % wtag, val_l1_loss,
                              self.global_step)

            # 3. Evaluate - test set - gt - testing on other datasets
            if eval_test and not self.test_loader == None:
                test_l2_loss, test_l1_loss = self.test_step(
                    device, writer, wtag)
                writer.add_scalar('Test_%s/L2_Loss' % wtag, test_l2_loss,
                                  self.global_step)
                writer.add_scalar('Test_%s/L1_Loss' % wtag, test_l1_loss,
                                  self.global_step)

            # Scheduler
            print(
                "[%d epoch] VAL [l2_loss %.2f   l1_loss %.2f]       TRAIN [l2_loss %.2f   l1_loss %.2f]"
                % (epoch, val_l2_loss, val_l1_loss, train_l2_loss,
                   train_l1_loss))
            self.scheduler.step(val_l2_loss)
            print("> LR", self.optimizer.param_groups[0]['lr'])
            writer.add_scalar('Train_%s/LR' % wtag,
                              self.optimizer.param_groups[0]['lr'],
                              self.global_step)

            self.global_step += 1

        # Save last epoch
        self.save_model(start_epoch + epoch + 1)

    def eval_detailed(self, device):
        self.net.eval()
        eval_l2_loss = []
        eval_l1_loss = []

        for batch in self.valid_loader:
            domain1, domain2_gt = batch
            assert domain1.shape[1] == self.net.module.n_channels
            assert domain2_gt.shape[1] == self.net.module.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                domain2_pred = self.net(domain1)
            loss_l2 = self.l2_detailed_eval(domain2_pred, domain2_gt)
            loss_l1 = self.l1_detailed_eval(domain2_pred, domain2_gt)

            eval_l2_loss += loss_l2.view(domain2_pred.shape[0],
                                         -1).mean(axis=1).data.cpu() * 100.
            eval_l1_loss += loss_l1.view(domain2_pred.shape[0],
                                         -1).mean(axis=1).data.cpu() * 100.

        return eval_l2_loss, eval_l1_loss

    ################ [Drop connections] ##################
    def drop_1hop_connections(edges_1hop, device, drop_version):
        valid_loaders = []
        for edge in edges_1hop:
            valid_loaders.append(iter(edge.valid_loader))

        correlations = torch.zeros(
            (len(edges_1hop) + 1, len(edges_1hop) + 1)).cuda()
        num_batches = len(valid_loaders[0])
        n_samples = 0
        for idx_batch in range(num_batches):
            domain2_1hop_ens_list = []

            for idx_edge, data_edge in enumerate(zip(edges_1hop,
                                                     valid_loaders)):
                edge, loader = data_edge
                domain1, domain2_exp_gt = next(loader)

                assert domain1.shape[1] == edge.net.module.n_channels
                assert domain2_exp_gt.shape[1] == edge.net.module.n_classes

                domain1 = domain1.to(device=device, dtype=torch.float32)
                domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                   dtype=torch.float32)

                with torch.no_grad():
                    one_hop_pred = edge.net(domain1)
                    domain2_1hop_ens_list.append(one_hop_pred.clone())
            # with_expert
            domain2_1hop_ens_list.append(domain2_exp_gt)

            # domain2_1hop_ens_list contains all data for current batch
            #import pdb
            #pdb.set_trace()
            correlations = utils.get_correlation_score(domain2_1hop_ens_list,
                                                       correlations,
                                                       drop_version)
            n_samples = n_samples + domain1.shape[0]
        correlations = correlations / n_samples
        return correlations

    ################ [1Hop Tests] ##################
    def eval_1hop_test(edges_1hop, loaders, l1_per_edge, l1_expert, writer,
                       wtag, save_idxes, device):
        with torch.no_grad():
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):

                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge

                    domain1, domain2_exp_gt, domain2_gt = next(loader)

                    assert domain1.shape[1] == edge.net.module.n_channels
                    assert domain2_exp_gt.shape[1] == edge.net.module.n_classes
                    assert domain2_gt.shape[1] == edge.net.module.n_classes

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)
                    domain2_gt = domain2_gt.to(device=device,
                                               dtype=torch.float32)

                    with torch.no_grad():
                        one_hop_pred = edge.net(domain1)

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=False)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(one_hop_pred[save_idxes]), 0)

                    l1_per_edge[idx_edge] += 100 * edge.l1(
                        one_hop_pred, domain2_gt).item()
                l1_expert += 100 * edge.l1(domain2_exp_gt, domain2_gt).item()
            l1_per_edge = np.array(l1_per_edge) / num_batches
            l1_expert = np.array(l1_expert) / num_batches
            return l1_per_edge, l1_expert, save_idxes, domain2_exp_gt, domain2_gt

    def eval_1hop_valid(edges_1hop, loaders, l1_per_edge, writer, wtag,
                        save_idxes, device):
        with torch.no_grad():
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):

                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt = next(loader)

                    assert domain1.shape[1] == edge.net.module.n_channels
                    assert domain2_exp_gt.shape[1] == edge.net.module.n_classes

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)

                    with torch.no_grad():
                        one_hop_pred = edge.net(domain1)

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=False)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(one_hop_pred[save_idxes]), 0)

                    l1_per_edge[idx_edge] += 100 * edge.l1(
                        one_hop_pred, domain2_exp_gt).item()
            l1_per_edge = np.array(l1_per_edge) / num_batches
            return l1_per_edge, save_idxes, domain2_exp_gt

    def eval_1hop(edges_1hop, save_idxes, save_idxes_test, device, writer,
                  valid_set_str, test_set_str, csv_results_path, epoch_str):
        wtag_valid = "to_%s_valid_set_%s" % (edges_1hop[0].expert2.identifier,
                                             valid_set_str)

        wtag_test = "to_%s_test_set_%s" % (edges_1hop[0].expert2.identifier,
                                           test_set_str)

        valid_loaders = []
        l1_per_edge = []
        for edge in edges_1hop:
            valid_loaders.append(iter(edge.valid_loader))
            l1_per_edge.append(0)

        start = time.time()
        l1_per_edge, save_idxes, domain2_exp_gt = Edge.eval_1hop_valid(
            edges_1hop, valid_loaders, l1_per_edge, writer, wtag_valid,
            save_idxes, device)
        end = time.time()
        print("time for VALID Edge.eval_1hop_valid", end - start)

        test_loaders = []
        test_edges = []
        l1_per_edge_test = []
        l1_expert_test = 0
        for edge in edges_1hop:
            if edge.test_loader != None:
                test_loaders.append(iter(edge.test_loader))
                test_edges.append(edge)
                l1_per_edge_test.append(0)

        start = time.time()
        if len(test_loaders) > 0:
            l1_per_edge_test, l1_expert_test, save_idxes_test, domain2_exp_gt_test, domain2_gt_test = Edge.eval_1hop_test(
                edges_1hop, test_loaders, l1_per_edge_test, l1_expert_test,
                writer, wtag_test, save_idxes_test, device)
        end = time.time()
        print("time for TEST Edge.eval_1hop_test", end - start)

        # valid db
        writer.add_images('%s/EXPERT' % (wtag_valid),
                          img_for_plot(domain2_exp_gt[save_idxes]), 0)

        # test db
        if len(test_loaders) > 0:
            writer.add_images(
                '%s/EXPERT' % (wtag_test),
                img_for_plot(domain2_exp_gt_test[save_idxes_test]), 0)
            writer.add_images('%s/GT' % (wtag_test),
                              img_for_plot(domain2_gt_test[save_idxes_test]),
                              0)

        tag = "to_%s" % (edges_1hop[0].expert2.identifier)
        print('%24s' % (tag))
        print('L1(expert, GT)_testset: %30.2f' % (l1_expert_test))

        print(
            "%24s  L1(directEdge, expert)_valset  L1(directEdge, GT)_testset" %
            (tag))

        # Show Individual Losses
        mean_l1_per_edge = np.mean(l1_per_edge)
        mean_l1_per_edge_test = 0
        if len(test_loaders) > 0:
            mean_l1_per_edge_test = np.mean(l1_per_edge_test)

        idx_test_edge = 0
        for idx_edge, edge in enumerate(edges_1hop):
            writer.add_scalar(
                '1hop_%s/L1_Loss_%s' % (wtag_valid, edge.expert1.identifier),
                l1_per_edge[idx_edge], 0)
            if edge.test_loader != None:
                writer.add_scalar(
                    '1hop_%s/L1_Loss_%s' %
                    (wtag_test, edge.expert1.identifier),
                    l1_per_edge_test[idx_test_edge], 0)
                print("Loss %19s: %30.2f   %30.2f" %
                      (edge.expert1.identifier, l1_per_edge[idx_edge],
                       l1_per_edge_test[idx_test_edge]))
                idx_test_edge = idx_test_edge + 1
            else:
                print("Loss %19s: %30.2f    %30s" %
                      (edge.expert1.identifier, l1_per_edge[idx_edge], '-'))
        print(
            "%25s-------------------------------------------------------------------------------------"
            % (" "))
        # print("Loss %-20s %30.2f   %30.2f" %
        #       ("average", mean_l1_per_edge, mean_l1_per_edge_test))

        # print("")
        print("")
        csv_path = os.path.join(
            csv_results_path, 'to__%s__valid_%s_test_%s_epoch_%s.csv' %
            (edges_1hop[0].expert2.identifier, valid_set_str, test_set_str,
             epoch_str))
        csv_file = open(csv_path, 'w')
        idx_test_edge = 0
        csv_file.write('model, dataset, src_domain, dst_domain,\n')
        for idx_edge, edge in enumerate(edges_1hop):
            csv_file.write('%s,%s,%s,%s,%10.6f\n' %
                           (epoch_str, valid_set_str, edge.expert1.identifier,
                            edge.expert2.identifier, l1_per_edge[idx_edge]))
            if edge.test_loader != None:
                csv_file.write(
                    '%s,%s,%s,%s,%10.6f\n' %
                    (epoch_str, test_set_str, edge.expert1.identifier,
                     edge.expert2.identifier, l1_per_edge_test[idx_test_edge]))
                idx_test_edge = idx_test_edge + 1
        if len(test_loaders) > 0:
            csv_file.write(
                '%s,%s,%s,%s,%10.6f\n' %
                (epoch_str, test_set_str, edges_1hop[0].expert2.identifier,
                 edges_1hop[0].expert2.identifier, l1_expert_test))
        csv_file.close()

        return save_idxes, save_idxes_test

    ################ [1Hop Ensembles] ##################
    def eval_1hop_ensemble_test_set(loaders, l1_per_edge, l1_ensemble1hop,
                                    l1_expert, edges_1hop, device, save_idxes,
                                    writer, wtag, edges_1hop_weights,
                                    ensemble_fct):
        with torch.no_grad():
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):
                domain2_1hop_ens_list = []

                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt, domain2_gt = next(loader)

                    assert domain1.shape[1] == edge.net.module.n_channels
                    assert domain2_exp_gt.shape[1] == edge.net.module.n_classes
                    assert domain2_gt.shape[1] == edge.net.module.n_classes

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)
                    domain2_gt = domain2_gt.to(device=device,
                                               dtype=torch.float32)

                    with torch.no_grad():
                        # Ensemble1Hop: 1hop preds
                        one_hop_pred = edge.net(domain1)
                        domain2_1hop_ens_list.append(one_hop_pred.clone())

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=False)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(one_hop_pred[save_idxes]), 0)
                        ''''
                        # save SSIM maps to tensorboard 
                        pred = one_hop_pred[save_idxes]
                        exp_res = domain2_exp_gt[save_idxes]
                        ssim_maps = utils.get_btw_tasks_ssim_score(
                            exp_res, pred)
                        ssim_maps = ssim_maps[:, 0, None, :, :]
                        pred = img_for_plot(pred)
                        ssim_maps = ssim_maps.repeat(1, pred.shape[1], 1, 1)
                        to_disp = torch.cat((pred, ssim_maps), 2)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/%s' % (wtag, edge.expert1.identifier), to_disp,
                            0)
                        '''

                    l1_per_edge[idx_edge] += 100 * edge.l1(
                        one_hop_pred, domain2_gt).item()

                domain2_1hop_ens_list.append(domain2_exp_gt)

                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)
                if not edge.ensemble_filter == None:
                    domain2_1hop_ens = edge.ensemble_filter(
                        domain2_1hop_ens_list.permute(1, 2, 3, 4, 0),
                        edge.expert2.domain_name)
                else:
                    domain2_1hop_ens = utils.combine_maps(
                        domain2_1hop_ens_list,
                        edges_1hop_weights,
                        edge.expert2.domain_name,
                        fct=ensemble_fct)
                l1_expert += 100 * edge.l1(domain2_exp_gt, domain2_gt).item()
                l1_ensemble1hop += 100 * edge.l1(domain2_1hop_ens,
                                                 domain2_gt).item()

            return l1_per_edge, l1_ensemble1hop, l1_expert, save_idxes, domain2_1hop_ens, domain2_exp_gt, domain2_gt, num_batches

    def eval_1hop_ensemble_valid_set(loaders, l1_per_edge, l1_ensemble1hop,
                                     edges_1hop, device, save_idxes, writer,
                                     wtag, edges_1hop_weights, ensemble_fct,
                                     config):
        with torch.no_grad():
            #crt_idx = 0
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):
                domain2_1hop_ens_list = []
                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt = next(loader)

                    assert domain1.shape[1] == edge.net.module.n_channels
                    assert domain2_exp_gt.shape[1] == edge.net.module.n_classes

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)

                    with torch.no_grad():
                        # Ensemble1Hop: 1hop preds
                        one_hop_pred = edge.net(domain1)
                        domain2_1hop_ens_list.append(one_hop_pred.clone())

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=False)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(one_hop_pred[save_idxes]), 0)

                    l1_per_edge[idx_edge] += 100 * edge.l1(
                        one_hop_pred, domain2_exp_gt).item()

                # with_expert
                domain2_1hop_ens_list.append(domain2_exp_gt)
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                if not edge.ensemble_filter == None:
                    domain2_1hop_ens = edge.ensemble_filter(
                        domain2_1hop_ens_list.permute(1, 2, 3, 4, 0),
                        edge.expert2.domain_name)
                else:
                    domain2_1hop_ens = utils.combine_maps(
                        domain2_1hop_ens_list, edges_1hop_weights,
                        edge.expert2.domain_name, ensemble_fct)
                '''
                # Save output for second iteration
                if config.getboolean('Training2Iters', 'train_2_iters'):
                    save_dir = "%s/%s" % (config.get(
                        'Training2Iters',
                        'save_next_iter_dir'), edge.expert2.identifier)

                    for elem_idx in range(domain2_1hop_ens.shape[0]):
                        save_path = "%s/%08d.npy" % (save_dir,
                                                     crt_idx + elem_idx)
                        np.save(save_path,
                                domain2_1hop_ens[elem_idx].data.cpu().numpy())
                    crt_idx += domain2_1hop_ens.shape[0]
                '''
                l1_ensemble1hop += 100 * edge.l1(domain2_1hop_ens,
                                                 domain2_exp_gt).item()
            return l1_per_edge, l1_ensemble1hop, save_idxes, domain2_1hop_ens, domain2_exp_gt, num_batches

    def eval_1hop_ensemble(edges_1hop, save_idxes, save_idxes_test, device,
                           writer, drop_version, edges_1hop_weights,
                           edges_1hop_test_weights, ensemble_fct, config):
        drop_str = 'with_drop' if drop_version >= 0 else 'no_drop'
        wtag_valid = "to_%s_valid_set_%s" % (edges_1hop[0].expert2.identifier,
                                             drop_str)
        wtag_test = "to_%s_test_set_%s" % (edges_1hop[0].expert2.identifier,
                                           drop_str)

        valid_loaders = []
        l1_per_edge = []
        l1_ensemble1hop = 0
        for edge in edges_1hop:
            valid_loaders.append(iter(edge.valid_loader))
            l1_per_edge.append(0)

        start = time.time()
        l1_per_edge, l1_ensemble1hop, save_idxes, domain2_1hop_ens, domain2_gt, num_batches = Edge.eval_1hop_ensemble_valid_set(
            valid_loaders, l1_per_edge, l1_ensemble1hop, edges_1hop, device,
            save_idxes, writer, wtag_valid, edges_1hop_weights, ensemble_fct,
            config)
        end = time.time()
        print("time for VALID Edge.eval_1hop_ensemble_aux", end - start)

        test_loaders = []
        test_edges = []
        l1_per_edge_test = []
        l1_ensemble1hop_test = 0
        l1_expert_test = 0
        for edge in edges_1hop:
            if edge.test_loader != None:
                test_loaders.append(iter(edge.test_loader))
                test_edges.append(edge)
                l1_per_edge_test.append(0)

        start = time.time()
        if len(test_loaders) > 0:
            l1_per_edge_test, l1_ensemble1hop_test, l1_expert_test, save_idxes_test, domain2_1hop_ens_test, domain2_exp_gt_test, domain2_gt_test, num_batches_test = Edge.eval_1hop_ensemble_test_set(
                test_loaders, l1_per_edge_test, l1_ensemble1hop_test,
                l1_expert_test, test_edges, device, save_idxes_test, writer,
                wtag_test, edges_1hop_test_weights, ensemble_fct)
        end = time.time()
        print("time for TEST Edge.eval_1hop_ensemble_test_set", end - start)

        # Show Ensemble
        writer.add_images('%s/ENSEMBLE' % (wtag_valid),
                          img_for_plot(domain2_1hop_ens[save_idxes]), 0)
        writer.add_images('%s/EXPERT' % (wtag_valid),
                          img_for_plot(domain2_gt[save_idxes]), 0)
        l1_ensemble1hop = np.array(l1_ensemble1hop) / num_batches
        writer.add_scalar('1hop_%s/L1_Loss_ensemble' % (wtag_valid),
                          l1_ensemble1hop, 0)

        # Show Ensemble - Test DB
        if len(test_loaders) > 0:
            writer.add_images(
                '%s/ENSEMBLE' % (wtag_test),
                img_for_plot(domain2_1hop_ens_test[save_idxes_test]), 0)
            writer.add_images(
                '%s/EXPERT' % (wtag_test),
                img_for_plot(domain2_exp_gt_test[save_idxes_test]), 0)
            writer.add_images('%s/GT' % (wtag_test),
                              img_for_plot(domain2_gt_test[save_idxes_test]),
                              0)
            l1_ensemble1hop_test = np.array(
                l1_ensemble1hop_test) / num_batches_test
            l1_expert_test = np.array(l1_expert_test) / num_batches_test
            writer.add_scalar('1hop_%s/L1_Loss_ensemble' % (wtag_test),
                              l1_ensemble1hop_test, 0)

        tag = "to_%s_%s" % (edges_1hop[0].expert2.identifier, drop_str)

        print("ensemble_fct:", ensemble_fct)
        print(
            "%24s  L1(ensemble_with_expert, Expert)_valset  L1(ensemble_with_expert, GT)_testset   L1(expert, GT)_testset"
            % (tag))

        print("Loss %19s: %30.2f   %30.2f %20.2f" %
              ("Ensemble1Hop", l1_ensemble1hop, l1_ensemble1hop_test,
               l1_expert_test))
        print(
            "%25s-------------------------------------------------------------------------------------"
            % (" "))

        # Show Individual Losses
        l1_per_edge = np.array(l1_per_edge) / num_batches
        mean_l1_per_edge = np.mean(l1_per_edge)
        mean_l1_per_edge_test = 0
        if len(test_loaders) > 0:
            l1_per_edge_test = np.array(l1_per_edge_test) / num_batches_test
            mean_l1_per_edge_test = np.mean(l1_per_edge_test)
        idx_test_edge = 0
        for idx_edge, edge in enumerate(edges_1hop):
            writer.add_scalar(
                '1hop_%s/L1_Loss_%s' % (wtag_valid, edge.expert1.identifier),
                l1_per_edge[idx_edge], 0)
            if edge.test_loader != None:
                writer.add_scalar(
                    '1hop_%s/L1_Loss_%s' %
                    (wtag_test, edge.expert1.identifier),
                    l1_per_edge_test[idx_test_edge], 0)
                print("Loss %19s: %30.2f   %30.2f" %
                      (edge.expert1.identifier, l1_per_edge[idx_edge],
                       l1_per_edge_test[idx_test_edge]))
                idx_test_edge = idx_test_edge + 1
            else:
                print("Loss %19s: %30.2f    %30s" %
                      (edge.expert1.identifier, l1_per_edge[idx_edge], '-'))
        print(
            "%25s-------------------------------------------------------------------------------------"
            % (" "))
        print("Loss %-20s %30.2f   %30.2f" %
              ("average", mean_l1_per_edge, mean_l1_per_edge_test))

        print("")
        print("")
        return save_idxes, save_idxes_test

    ######## Save 1hop ensembles ###############
    def save_1hop_ensemble_next_iter_set(loaders, edges_1hop, device,
                                         ensemble_fct, config, save_dir):
        with torch.no_grad():
            crt_idx = 0
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):
                domain2_1hop_ens_list = []
                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt = next(loader)

                    assert domain1.shape[1] == edge.net.module.n_channels
                    assert domain2_exp_gt.shape[1] == edge.net.module.n_classes

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)

                    with torch.no_grad():
                        # Ensemble1Hop: 1hop preds
                        one_hop_pred = edge.net(domain1)
                        domain2_1hop_ens_list.append(one_hop_pred.clone())

                # with_expert
                domain2_1hop_ens_list.append(domain2_exp_gt)
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                if not edge.ensemble_filter == None:
                    domain2_1hop_ens = edge.ensemble_filter(
                        domain2_1hop_ens_list.permute(1, 2, 3, 4, 0),
                        edge.expert2.domain_name)
                else:
                    domain2_1hop_ens = utils.combine_maps(
                        domain2_1hop_ens_list, [], edge.expert2.domain_name,
                        ensemble_fct)

                save_dir_ = os.path.join(save_dir, edge.expert2.identifier)
                for elem_idx in range(domain2_1hop_ens.shape[0]):
                    save_path = "%s/%08d.npy" % (save_dir_, crt_idx + elem_idx)
                    np.save(save_path,
                            domain2_1hop_ens[elem_idx].data.cpu().numpy())
                crt_idx += domain2_1hop_ens.shape[0]

            if num_batches > 0 and config.getboolean('Training2Iters',
                                                     'train_2_iters'):
                print("[Iter2] Supervision Saved to:", save_dir_)

    def save_1hop_ensemble(edges_1hop, device, ensemble_fct, config):
        next_iter_loaders = []
        test_no_gt_loaders = []
        for edge in edges_1hop:
            next_iter_loaders.append(iter(edge.next_iter_loader))
            test_no_gt_loaders.append(iter(edge.test_no_gt_loader))

        start = time.time()
        save_dir = os.path.join(
            config.get('Training2Iters', 'NEXT_ITER_DST_TRAIN_PATH'),
            config.get('Training2Iters', 'NEXT_ITER_DB_PATH'))
        Edge.save_1hop_ensemble_next_iter_set(next_iter_loaders, edges_1hop,
                                              device, ensemble_fct, config,
                                              save_dir)
        end = time.time()
        print("time for NEXT ITER SET Edge.save_1hop_ensemble_next_iter_set",
              end - start)
        if config.getint('Training2Iters', 'iter_2_src_data') == 2:
            start = time.time()
            save_dir = os.path.join(
                config.get('Training2Iters', 'ENSEMBLE_OUTPUT_PATH_TEST'),
                config.get('Paths', 'TEST_PATH'))
            Edge.save_1hop_ensemble_next_iter_set(test_no_gt_loaders,
                                                  edges_1hop, device,
                                                  ensemble_fct, config,
                                                  save_dir)
            end = time.time()
            print("time for TEST SET Edge.save_1hop_ensemble_next_iter_set",
                  end - start)

        return

    def ensemble_histogram(edges_loaders_1hop, end_id, device):
        with torch.no_grad():
            num_batches = len(edges_loaders_1hop[0][1])
            for idx_batch in range(num_batches):
                domain2_1hop_ens_list = []
                for idx_edge, data_edge in enumerate(edges_loaders_1hop):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt = next(loader)

                    assert domain1.shape[1] == edge.net.module.n_channels
                    assert domain2_exp_gt.shape[1] == edge.net.module.n_classes

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)

                    # Ensemble1Hop: 1hop preds
                    one_hop_pred = edge.net(domain1)
                    domain2_1hop_ens_list.append(one_hop_pred.clone())

                # add direct expert prediction to the ensemble
                domain2_1hop_ens_list.append(domain2_exp_gt)
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                histograms = utils.pixels_histogram(domain2_1hop_ens_list,
                                                    end_id)

            return

    ################ [2Hop Ensembles] ##################
    # def train_from_2hops_ens_step(self, device, edges_2hop):
    #     self.net.train()

    #     train_l1_loss = 0
    #     train_l1_loss_ensemble = 0
    #     train_l2_loss = 0
    #     tag = 'Train_2hop/'
    #     # tag = 'Train_2hop/%s---%s' % (self.expert1.str_id, self.expert2.str_id)

    #     for idx_batch, batch in enumerate(self.train_loader):
    #         domain1, domain2_gt = batch
    #         assert domain1.shape[1] == self.net.n_channels
    #         assert domain2_gt.shape[1] == self.net.n_classes

    #         domain1 = domain1.to(device=device, dtype=torch.float32)
    #         domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

    #         # 1hop = direct prediction
    #         domain2_pred = self.net(domain1)

    #         # pseudo-gt = average over all 2hop predictions
    #         domain2_pseudo_gt = torch.zeros_like(domain2_gt)
    #         for edge1, edge2 in edges_2hop:
    #             composed_out = edge2.net(edge1.net(domain1))
    #             domain2_pseudo_gt += composed_out

    #             if idx_batch == len(self.train_loader) - 1:
    #                 # log all
    #                 experts_names = "%s-%s-%s" % (
    #                     edge1.expert1.__class__.__name__,
    #                     edge1.expert2.__class__.__name__,
    #                     edge2.expert2.__class__.__name__)
    #                 self.writer.add_images('%s/%s' % (tag, experts_names),
    #                                        img_for_plot(composed_out[:3]),
    #                                        self.global_step)

    #         domain2_pseudo_gt /= len(edges_2hop)

    #         l2_loss = self.l2(domain2_pred, domain2_pseudo_gt)
    #         train_l2_loss += l2_loss.item()

    #         with torch.no_grad():
    #             train_l1_loss_ensemble += self.l1(domain2_pseudo_gt,
    #                                               domain2_gt).item()
    #             train_l1_loss += self.l1(domain2_pred, domain2_gt).item()

    #         # Optimizer
    #         self.optimizer.zero_grad()
    #         l2_loss.backward()
    #         self.optimizer.step()

    #     self.writer.add_images('%s/Input' % (tag), img_for_plot(domain1[:3]),
    #                            self.global_step)

    #     self.writer.add_images('%s/Expert0HopOut' % (tag),
    #                            img_for_plot(domain2_gt[:3]), self.global_step)

    #     self.writer.add_images('%s/Ensemble2HopOut' % (tag),
    #                            img_for_plot(domain2_pseudo_gt[:3]),
    #                            self.global_step)

    #     self.writer.add_images('%s/Edge1HopOut' % (tag),
    #                            img_for_plot(domain2_pred[:3]),
    #                            self.global_step)
    #     return train_l2_loss / len(
    #         self.train_loader) * 100, train_l1_loss / len(
    #             self.train_loader) * 100, train_l1_loss_ensemble / len(
    #                 self.train_loader) * 100

    def eval_from_2hops_ens_step(self, device, edges_2hop, writer,
                                 use_expert_gt):
        self.net.eval()
        eval_l2_loss = 0
        eval_l1_loss = 0
        eval_l1_loss_ensemble = 0
        tag = 'Valid_2hop_%s_%s' % (self.expert1.identifier,
                                    self.expert2.identifier)

        for idx_batch, batch in enumerate(self.valid_loader):
            domain1, domain2_expertgt = batch
            assert domain1.shape[1] == self.net.module.n_channels
            assert domain2_expertgt.shape[1] == self.net.module.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_expertgt = domain2_expertgt.to(device=device,
                                                   dtype=torch.float32)

            with torch.no_grad():
                # 1hop = direct prediction
                domain2_pred = self.net(domain1)

                # Ensemble2Hop: over all 2hop preds
                domain2_2hop_ens = domain2_pred.clone()
                no_add_terms = 1

            if use_expert_gt:
                domain2_2hop_ens += domain2_expertgt
                no_add_terms += 1

            for edge1, edge2 in edges_2hop:
                composed_out = edge2.net(edge1.net(domain1))
                domain2_2hop_ens += composed_out

                if idx_batch == len(self.valid_loader) - 1:
                    rnd_idx = np.random.choice(domain1.shape[0],
                                               size=(3),
                                               replace=False)
                    # Log
                    experts_names = "%s-%s-%s" % (edge1.expert1.identifier,
                                                  edge1.expert2.identifier,
                                                  edge2.expert2.identifier)

                    writer.add_images('%s/%s' % (tag, experts_names),
                                      img_for_plot(composed_out[rnd_idx]),
                                      self.global_step)

            domain2_2hop_ens /= (len(edges_2hop) + no_add_terms)

            eval_l2_loss += self.l2(domain2_pred, domain2_2hop_ens).item()
            eval_l1_loss += self.l1(domain2_pred, domain2_expertgt).item()
            if len(edges_2hop) > 0:
                eval_l1_loss_ensemble += self.l1(domain2_2hop_ens,
                                                 domain2_expertgt).item()

        # Log
        writer.add_images('%s/Input' % tag, img_for_plot(domain1[rnd_idx]),
                          self.global_step)

        writer.add_images('%s/EXPERT' % tag,
                          img_for_plot(domain2_expertgt[rnd_idx]),
                          self.global_step)

        writer.add_images('%s/EDGE' % tag, img_for_plot(domain2_pred[rnd_idx]),
                          self.global_step)

        writer.add_images('%s/ENSEMBLE_2HOP' % (tag),
                          img_for_plot(domain2_2hop_ens[rnd_idx]),
                          self.global_step)

        return eval_l2_loss / len(self.valid_loader) * 100, eval_l1_loss / len(
            self.valid_loader) * 100, eval_l1_loss_ensemble / len(
                self.valid_loader) * 100

    def train_from_2hops_ens(self, graph, epochs, drop_version, device, writer,
                             use_expert_gt):
        wtag = '%s_%s' % (self.expert1.identifier, self.expert2.identifier)
        start_id = self.expert1.identifier
        end_id = self.expert2.identifier
        edges_2hop = []
        edges_1hop = []

        print("Direct Edge X->Y: %s" % (self))
        for edge_xk in graph.edges:
            if edge_xk.ill_posed:
                continue
            if edge_xk.expert1.identifier == start_id:
                if edge_xk.expert2.identifier == end_id:
                    continue

                for edge_ky in graph.edges:
                    if edge_ky.expert1.identifier == edge_xk.expert2.identifier and edge_ky.expert2.identifier == end_id:
                        if edge_ky.ill_posed:
                            continue
                        print("\tAdding 2hop: %s, %s" % (edge_xk, edge_ky))
                        edges_2hop.append((edge_xk, edge_ky))
            else:
                if edge_xk.expert2.identifier == end_id:
                    edges_1hop.append(edge_xk)

        if len(edges_2hop) < 1:
            return

        for epoch in range(epochs):
            # train_l2_loss, train_l1_loss, train_l1_loss_ensemble = self.train_from_2hops_step(
            #     device, edges_2hop)
            # self.writer.add_scalar('Train_2hop/L2_Loss', train_l2_loss,
            #                        self.global_step)
            # self.writer.add_scalar('Train_2hop/L1_Loss', train_l1_loss,
            #                        self.global_step)
            # self.writer.add_scalar('Train_2hop/L1_Loss_ensemble',
            #                        train_l1_loss_ensemble, self.global_step)

            val_l2_loss, val_l1_loss, val_l1_loss_ensemble = self.eval_from_2hops_ens_step(
                device, edges_2hop, writer=writer, use_expert_gt=use_expert_gt)
            writer.add_scalar('Valid_2hop_%s/L2_Loss' % wtag, val_l2_loss,
                              self.global_step)

            if val_l1_loss_ensemble > 0:
                writer.add_scalar('Valid_2hop_%s/L1_Loss' % wtag, val_l1_loss,
                                  self.global_step)
                writer.add_scalar('Valid_2hop_%s/L1_Loss_ensemble' % wtag,
                                  val_l1_loss_ensemble, self.global_step)
                writer.add_scalar(
                    'Valid_2hop_%s/Delta_L1_EdgeMinusEnsemble_Pozitive_is_Better'
                    % wtag, val_l1_loss - val_l1_loss_ensemble,
                    self.global_step)

            # Scheduler
            self.scheduler.step(val_l2_loss)
            writer.add_scalar('Train_2hop_%s/LR' % wtag,
                              self.optimizer.param_groups[0]['lr'],
                              self.global_step)

            self.global_step += 1
