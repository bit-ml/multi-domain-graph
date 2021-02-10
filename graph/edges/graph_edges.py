import glob
import logging
import os
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
#from graph.edges.dataset2d import (Domain2DDataset, DomainTestDataset,
#                                   DomainTrainNextIterDataset)
from graph.edges.dataset2d import ImageLevelDataset
from graph.edges.unet.unet_model import UNetGood
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import utils
from utils.utils import img_for_plot
from utils.utils import EnsembleFilter_TwdExpert


class Edge:
    def __init__(self, config, expert1, expert2, device, rnd_sampler, silent,
                 valid_shuffle, iter_no, bs_train, bs_test):
        super(Edge, self).__init__()
        self.config = config
        self.silent = silent

        # Initialize ensemble model for destination task
        similarity_fct = config.get('Ensemble', 'similarity_fct')
        self.ensemble_filter = EnsembleFilter_TwdExpert(
            n_channels=expert2.n_maps,
            similarity_fct=similarity_fct,
            threshold=0.5)
        #self.ensemble_filter = nn.DataParallel(self.ensemble_filter)

        self.init_edge(expert1, expert2, device)
        self.init_loaders(bs=bs_train * torch.cuda.device_count(),
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
        nesterov = config.getboolean('Training', 'nesterov')

        self.lr = learning_rate
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(),
                                       lr=self.lr,
                                       weight_decay=weight_decay,
                                       nesterov=nesterov,
                                       momentum=momentum)
        elif optimizer_type[:4] == "adam":
            amsgrad = config.getboolean('Training', 'amsgrad')
            optimizer_class = optim.Adam
            if optimizer_type == 'adamw':
                optimizer_class = optim.AdamW

            self.optimizer = optimizer_class(self.net.parameters(),
                                             lr=self.lr,
                                             weight_decay=weight_decay,
                                             amsgrad=amsgrad)
        else:
            print("Incorrect optimizer", optimizer_type)
            sys.exit(-1)

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

        if config.getboolean('Edge Models', 'save_models'):
            self.save_model_dir = os.path.join(
                config.get('Edge Models', 'save_path'),
                config.get('Run id', 'datetime'),
                '%s_%s' % (expert1.identifier, expert2.identifier))

            if not os.path.exists(self.save_model_dir):
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

        self.net = UNetGood(n_channels=self.expert1.get_n_final_maps(),
                            n_classes=self.expert2.n_maps,
                            from_exp=expert1,
                            to_exp=expert2).to(device)

        self.net = nn.DataParallel(self.net)
        # total_params = sum(p.numel() for p in self.net.parameters()) / 1e+6
        # trainable_params = sum(p.numel() for p in self.net.parameters()) / 1e+6
        # print("\tNumber of parameters %.2fM (Trainable %.2fM)" %
        #       (total_params, trainable_params))

    def copy_model(self, device):
        self.prev_net = UNetGood(n_channels=self.expert1.n_maps,
                                 n_classes=self.expert2.n_maps,
                                 bilinear=True).to(device)
        self.prev_net = nn.DataParallel(self.prev_net)
        self.prev_net.load_state_dict(self.net.state_dict())

    def init_loaders(self, bs, bs_test, n_workers, rnd_sampler, valid_shuffle,
                     iter_no):
        # Load train dataset
        train_ds = ImageLevelDataset(self.expert1, self.expert2, self.config,
                                     iter_no, 'TRAIN')
        print("\tTrain ds", len(train_ds), "==========")
        self.train_loader = DataLoader(train_ds,
                                       batch_size=bs,
                                       shuffle=True,
                                       num_workers=n_workers)

        # Load valid dataset
        valid_ds = ImageLevelDataset(self.expert1, self.expert2, self.config,
                                     iter_no, 'VALID')
        print("\tValid ds", len(valid_ds), "==========")
        self.valid_loader = DataLoader(valid_ds,
                                       batch_size=bs_test,
                                       shuffle=valid_shuffle,
                                       num_workers=n_workers)
        # Load test dataset
        test_ds = ImageLevelDataset(self.expert1, self.expert2, self.config,
                                    iter_no, 'TEST')
        print("\tTest ds", len(test_ds), "==========")
        if len(test_ds) > 0:
            self.test_loader = DataLoader(test_ds,
                                          batch_size=bs_test,
                                          shuffle=False,
                                          num_workers=n_workers)
        else:
            self.test_loader = None

        if iter_no + 1 <= self.config.getint('General', 'n_iters'):
            n_next_iter_train_subsets = len(
                self.config.get('PathsIter%d' % (iter_no + 1),
                                'ITER%d_TRAIN_SRC_PATH' %
                                (iter_no + 1)).split('\n'))
            n_next_iter_valid_subsets = len(
                self.config.get('PathsIter%d' % (iter_no + 1),
                                'ITER%d_VALID_SRC_PATH' %
                                (iter_no + 1)).split('\n'))
            n_next_iter_test_subsets = len(
                self.config.get('PathsIter%d' % (iter_no + 1),
                                'ITER%d_TEST_SRC_PATH' %
                                (iter_no + 1)).split('\n'))
            self.next_iter_train_loaders = []
            for i in range(n_next_iter_train_subsets):
                next_iter_train_ds = ImageLevelDataset(
                    self.expert1,
                    self.expert2,
                    self.config,
                    iter_no + 1,
                    'TRAIN',
                    for_next_iter=True,
                    for_next_iter_idx_subset=i)
                print("\tNext iter train ds - subset %d" % i,
                      len(next_iter_train_ds), "==========")
                self.next_iter_train_loaders.append(
                    DataLoader(next_iter_train_ds,
                               batch_size=bs_test,
                               shuffle=False,
                               num_workers=n_workers))
            self.next_iter_valid_loaders = []
            for i in range(n_next_iter_valid_subsets):
                next_iter_valid_ds = ImageLevelDataset(
                    self.expert1,
                    self.expert2,
                    self.config,
                    iter_no + 1,
                    'VALID',
                    for_next_iter=True,
                    for_next_iter_idx_subset=i)
                print("\tNext iter valid ds - subset %d" % i,
                      len(next_iter_valid_ds), "==========")
                self.next_iter_valid_loaders.append(
                    DataLoader(next_iter_valid_ds,
                               batch_size=bs_test,
                               shuffle=False,
                               num_workers=n_workers))

            self.next_iter_test_loaders = []
            for i in range(n_next_iter_test_subsets):
                next_iter_test_ds = ImageLevelDataset(
                    self.expert1,
                    self.expert2,
                    self.config,
                    iter_no + 1,
                    'TEST',
                    for_next_iter=True,
                    for_next_iter_idx_subset=i)
                print("\tNext iter test ds - subset %d" % i,
                      len(next_iter_test_ds), "==========")
                self.next_iter_test_loaders.append(
                    DataLoader(next_iter_test_ds,
                               batch_size=bs_test,
                               shuffle=False,
                               num_workers=n_workers))

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

        writer.add_images('Train_%s/Input' % (wtag),
                          img_for_plot(domain1[:3], self.expert1.identifier),
                          self.global_step)

        writer.add_images(
            'Train_%s/GT_EXPERT' % (wtag),
            img_for_plot(domain2_gt[:3], self.expert2.identifier),
            self.global_step)

        writer.add_images(
            'Train_%s/Output' % (wtag),
            img_for_plot(domain2_pred[:3], self.expert2.identifier),
            self.global_step)

        return train_l2_loss / len(
            self.train_loader) * 100, train_l1_loss / len(
                self.train_loader) * 100

    def eval_step(self, device, writer, wtag):
        self.net.eval()
        eval_l2_loss = 0
        eval_l1_loss = 0

        for batch in self.valid_loader:
            domain1, domain2_gt = batch

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                domain2_pred = self.net(domain1).clamp(min=0, max=1)
            l2_loss = self.l2(domain2_pred, domain2_gt)
            l1_loss = self.l1(domain2_pred, domain2_gt)

            eval_l2_loss += l2_loss.item()
            eval_l1_loss += l1_loss.item()

        writer.add_images('Valid_%s/Input' % wtag,
                          img_for_plot(domain1[:3], self.expert1.identifier),
                          self.global_step)

        writer.add_images(
            'Valid_%s/GT_EXPERT' % wtag,
            img_for_plot(domain2_gt[:3], self.expert2.identifier),
            self.global_step)

        writer.add_images(
            'Valid_%s/Output' % wtag,
            img_for_plot(domain2_pred[:3], self.expert2.identifier),
            self.global_step)

        return eval_l2_loss / len(self.valid_loader) * 100, eval_l1_loss / len(
            self.valid_loader) * 100

    # def test_step(self, device, writer, wtag):
    #     """Currently should work as eval_step
    #     """
    #     self.net.eval()
    #     test_l2_loss = 0
    #     test_l1_loss = 0

    #     for batch in self.test_loader:
    #         domain1, domain2_gt, _ = batch

    #         domain1 = domain1.to(device=device, dtype=torch.float32)
    #         domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

    #         with torch.no_grad():
    #             domain2_pred = self.net(domain1).clamp(min=0, max=1)
    #         l2_loss = self.l2(domain2_pred, domain2_gt)
    #         l1_loss = self.l1(domain2_pred, domain2_gt)

    #         test_l2_loss += l2_loss.item()
    #         test_l1_loss += l1_loss.item()

    #     writer.add_images('Test_%s/Input' % wtag, img_for_plot(domain1[:3]),
    #                       self.global_step)

    #     writer.add_images('Test_%s/GT' % wtag, img_for_plot(domain2_gt[:3]),
    #                       self.global_step)

    #     writer.add_images('Test_%s/Output' % wtag,
    #                       img_for_plot(domain2_pred[:3]), self.global_step)

    #     return test_l2_loss / len(self.test_loader) * 100, test_l1_loss / len(
    #         self.test_loader) * 100

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
            self.scheduler.step(val_l1_loss + val_l2_loss)
            print("> LR", self.optimizer.param_groups[0]['lr'])
            writer.add_scalar('Train_%s/LR' % wtag,
                              self.optimizer.param_groups[0]['lr'],
                              self.global_step)

            self.global_step += 1

        # Save last epoch
        self.save_model(start_epoch + epoch + 1)

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
                            img_for_plot(one_hop_pred[save_idxes], edge.expert2.identifier), 0)

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
                            img_for_plot(one_hop_pred[save_idxes], edge.expert2.identifier), 0)

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
                          img_for_plot(domain2_exp_gt[save_idxes], edge.expert2.identifier), 0)

        # test db
        if len(test_loaders) > 0:
            writer.add_images(
                '%s/EXPERT' % (wtag_test),
                img_for_plot(domain2_exp_gt_test[save_idxes_test], edge.expert2.identifier), 0)
            writer.add_images('%s/GT' % (wtag_test),
                              img_for_plot(domain2_gt_test[save_idxes_test], edge.expert2.identifier),
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
                                    writer, wtag, edges_1hop_weights):
        with torch.no_grad():
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):
                domain2_1hop_ens_list = []

                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt, domain2_gt = next(loader)

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)
                    domain2_gt = domain2_gt.to(device=device,
                                               dtype=torch.float32)

                    with torch.no_grad():
                        # Ensemble1Hop: 1hop preds
                        one_hop_pred = edge.net(domain1).clamp(min=0, max=1)
                        domain2_1hop_ens_list.append(one_hop_pred.clone())

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=False)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(one_hop_pred[save_idxes],
                                         edge.expert2.identifier), 0)
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

                domain2_1hop_ens = edge.ensemble_filter(
                    domain2_1hop_ens_list.permute(1, 2, 3, 4, 0),
                    edge.expert2.domain_name)

                l1_expert += 100 * edge.l1(domain2_exp_gt, domain2_gt).item()
                l1_ensemble1hop += 100 * edge.l1(domain2_1hop_ens,
                                                 domain2_gt).item()

            return l1_per_edge, l1_ensemble1hop, l1_expert, save_idxes, domain2_1hop_ens, domain2_exp_gt, domain2_gt, num_batches

    def eval_1hop_ensemble_valid_set(loaders, l1_per_edge, l1_ensemble1hop,
                                     edges_1hop, device, save_idxes, writer,
                                     wtag, edges_1hop_weights, config):
        with torch.no_grad():
            #crt_idx = 0
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):
                domain2_1hop_ens_list = []
                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt = next(loader)

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)

                    with torch.no_grad():
                        # Ensemble1Hop: 1hop preds
                        one_hop_pred = edge.net(domain1).clamp(min=0, max=1)
                        domain2_1hop_ens_list.append(one_hop_pred.clone())

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=True)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(one_hop_pred[save_idxes],
                                         edge.expert2.identifier), 0)

                    l1_per_edge[idx_edge] += 100 * edge.l1(
                        one_hop_pred, domain2_exp_gt).item()

                # with_expert
                domain2_1hop_ens_list.append(domain2_exp_gt)
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                domain2_1hop_ens = edge.ensemble_filter(
                    domain2_1hop_ens_list.permute(1, 2, 3, 4, 0),
                    edge.expert2.domain_name)
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
                           edges_1hop_test_weights, config):
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
            save_idxes, writer, wtag_valid, edges_1hop_weights, config)
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
                wtag_test, edges_1hop_test_weights)
        end = time.time()
        print("time for TEST Edge.eval_1hop_ensemble_test_set", end - start)

        # Show Ensemble
        writer.add_images(
            '%s/ENSEMBLE' % (wtag_valid),
            img_for_plot(domain2_1hop_ens[save_idxes],
                         edges_1hop[0].expert2.identifier), 0)
        writer.add_images(
            '%s/EXPERT' % (wtag_valid),
            img_for_plot(domain2_gt[save_idxes],
                         edges_1hop[0].expert2.identifier), 0)
        l1_ensemble1hop = np.array(l1_ensemble1hop) / num_batches
        writer.add_scalar('1hop_%s/L1_Loss_ensemble' % (wtag_valid),
                          l1_ensemble1hop, 0)

        # Show Ensemble - Test DB
        if len(test_loaders) > 0:
            writer.add_images(
                '%s/ENSEMBLE' % (wtag_test),
                img_for_plot(domain2_1hop_ens_test[save_idxes_test],
                             edges_1hop[0].expert2.identifier), 0)
            writer.add_images(
                '%s/EXPERT' % (wtag_test),
                img_for_plot(domain2_exp_gt_test[save_idxes_test],
                             edges_1hop[0].expert2.identifier), 0)
            # from PIL import Image
            # pred_logits = domain2_exp_gt_test[1]
            # rgb_img = (pred_logits * 255.).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)

            # Image.fromarray(rgb_img).save("t_initial.png")
            # aux = 2 * (pred_logits - 0.5)
            # aux[2, :, :] = 0
            # aux_norm = aux.norm(dim=0, keepdim=True)
            # aux_renormed = aux / aux_norm
            # # transform it back to RGB
            # normals_maps = 0.5 * aux_renormed + 0.5

            # rgb_img = (normals_maps * 255.).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
            # Image.fromarray(rgb_img).save("t_modificat012.png")
            # rgb_img = (normals_maps[[0, 2, 1]] * 255.).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
            # Image.fromarray(rgb_img).save("t_modificat021.png")
            # rgb_img = (normals_maps[[1, 2, 0]] * 255.).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
            # Image.fromarray(rgb_img).save("t_modificat120.png")
            # rgb_img = (normals_maps[[1, 0, 2]] * 255.).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
            # Image.fromarray(rgb_img).save("t_modificat102.png")
            # rgb_img = (normals_maps[[2, 0, 1]] * 255.).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
            # Image.fromarray(rgb_img).save("t_modificat201.png")
            # rgb_img = (normals_maps[[2, 1, 0]] * 255.).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
            # Image.fromarray(rgb_img).save("t_modificat210.png")
            writer.add_images(
                '%s/GT' % (wtag_test),
                img_for_plot(domain2_gt_test[save_idxes_test],
                             edges_1hop[0].expert2.identifier), 0)
            l1_ensemble1hop_test = np.array(
                l1_ensemble1hop_test) / num_batches_test
            l1_expert_test = np.array(l1_expert_test) / num_batches_test
            writer.add_scalar('1hop_%s/L1_Loss_ensemble' % (wtag_test),
                              l1_ensemble1hop_test, 0)

        tag = "to_%s_%s" % (edges_1hop[0].expert2.identifier, drop_str)
        print("load_path", config.get('Edge Models', 'load_path'))
        print("Ensemble - sim fct: ", config.get('Ensemble', 'similarity_fct'))
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
    def save_1hop_ensemble_next_iter_set(loaders, edges_1hop, device, config,
                                         save_dir):
        with torch.no_grad():
            crt_idx = 0
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):
                domain2_1hop_ens_list = []
                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt = next(loader)

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)

                    with torch.no_grad():
                        # Ensemble1Hop: 1hop preds
                        one_hop_pred = edge.net(domain1).clamp(min=0, max=1)
                        domain2_1hop_ens_list.append(one_hop_pred.clone())

                # with_expert
                domain2_1hop_ens_list.append(domain2_exp_gt)
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                domain2_1hop_ens = edge.ensemble_filter(
                    domain2_1hop_ens_list.permute(1, 2, 3, 4, 0),
                    edge.expert2.domain_name)

                save_dir_ = os.path.join(save_dir, edge.expert2.identifier)
                for elem_idx in range(domain2_1hop_ens.shape[0]):
                    save_path = "%s/%08d.npy" % (save_dir_, crt_idx + elem_idx)
                    np.save(save_path,
                            domain2_1hop_ens[elem_idx].data.cpu().numpy())
                crt_idx += domain2_1hop_ens.shape[0]

            if num_batches > 0:
                print("[Iter2] Supervision Saved to:", save_dir_)

    def save_1hop_ensemble(edges_1hop, device, config, iter_no):
        next_iter_train_store_paths = config.get(
            'PathsIter%d' % (iter_no + 1),
            'ITER%d_TRAIN_STORE_PATH' % (iter_no + 1)).split('\n')
        next_iter_valid_store_paths = config.get(
            'PathsIter%d' % (iter_no + 1),
            'ITER%d_VALID_STORE_PATH' % (iter_no + 1)).split('\n')
        next_iter_test_store_paths = config.get(
            'PathsIter%d' % (iter_no + 1),
            'ITER%d_TEST_STORE_PATH' % (iter_no + 1)).split('\n')

        # Save next iter train subsets
        for i in range(len(next_iter_train_store_paths)):
            local_next_iter_train_loaders = []
            for edge in edges_1hop:
                local_next_iter_train_loaders.append(
                    iter(edge.next_iter_train_loaders[i]))
            Edge.save_1hop_ensemble_next_iter_set(
                local_next_iter_train_loaders, edges_1hop, device, config,
                next_iter_train_store_paths[i])

        for i in range(len(next_iter_valid_store_paths)):
            local_next_iter_valid_loaders = []
            for edge in edges_1hop:
                local_next_iter_valid_loaders.append(
                    iter(edge.next_iter_valid_loaders[i]))
            Edge.save_1hop_ensemble_next_iter_set(
                local_next_iter_valid_loaders, edges_1hop, device, config,
                next_iter_valid_store_paths[i])

        for i in range(len(next_iter_test_store_paths)):
            local_next_iter_test_loaders = []
            for edge in edges_1hop:
                local_next_iter_test_loaders.append(
                    iter(edge.next_iter_test_loaders[i]))
            Edge.save_1hop_ensemble_next_iter_set(
                local_next_iter_test_loaders, edges_1hop, device, config,
                next_iter_test_store_paths[i])

        return
