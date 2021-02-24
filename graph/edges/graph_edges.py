import collections
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
from experts.basic_expert import BasicExpert
from graph.edges.dataset2d import ImageLevelDataset
from graph.edges.unet.unet_model import UNetGood
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import utils
from utils.utils import EnsembleFilter_TwdExpert, img_for_plot


def labels_to_multichan(inp_1chan_cls, n_classes):
    bs, h, w = inp_1chan_cls.shape

    outp_multichan = torch.zeros(
        (bs, n_classes, h, w)).to(inp_1chan_cls.device).float()
    for chan in range(n_classes):
        outp_multichan[:, chan][inp_1chan_cls == chan] = 1.
    return outp_multichan


class Edge:
    def __init__(self, config, expert1, expert2, device, rnd_sampler, silent,
                 valid_shuffle, iter_no, bs_train, bs_test):
        super(Edge, self).__init__()
        self.config = config
        self.silent = silent

        # Initialize ensemble model for destination task
        similarity_fct = config.get('Ensemble', 'similarity_fct')
        self.ensemble_filter = EnsembleFilter_TwdExpert(
            n_channels=expert2.no_maps_as_ens_input(),
            similarity_fct=similarity_fct,
            normalize_output_fcn=expert2.postprocess_eval,
            threshold=0.5,
            dst_domain_name=expert2.domain_name)
        #self.ensemble_filter = nn.DataParallel(self.ensemble_filter)

        self.init_edge(expert1, expert2, device)

        test1 = config.get('General', 'Steps_Iter1_test')
        test2 = config.get('General', 'Steps_Iter2_test')
        if test1 or test2:
            n_workers = 8
        else:
            n_workers = max(8, 7 * torch.cuda.device_count())

        self.init_loaders(bs=bs_train * torch.cuda.device_count(),
                          bs_test=bs_test * torch.cuda.device_count(),
                          n_workers=n_workers,
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

        if self.expert2.get_task_type() == BasicExpert.TASK_CLASSIFICATION:
            self.training_losses = [
                nn.CrossEntropyLoss(weight=self.expert2.classification_weights)
            ]
            self.gt_transform = (lambda x: x.squeeze(1).long())
            self.to_ens_transform = labels_to_multichan
        else:
            self.training_losses = [nn.SmoothL1Loss(beta=0.8)]
            self.eval_losses = [nn.L1Loss()]
            self.gt_transform = (lambda x: x)
            self.to_ens_transform = (lambda x, y: x)

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

    def init_edge(self, expert1, expert2, device):
        self.expert1 = expert1
        self.expert2 = expert2
        self.name = "%s -> %s" % (expert1.identifier, expert2.identifier)

        net = UNetGood(n_channels=expert1.no_maps_as_nn_input(),
                       n_classes=expert2.no_maps_as_nn_output(),
                       from_exp=expert1,
                       to_exp=expert2).to(device)
        self.net = nn.DataParallel(net)

        total_params = sum(p.numel() for p in self.net.parameters()) / 1e+6
        trainable_params = sum(p.numel() for p in self.net.parameters()) / 1e+6
        print("\tNumber of parameters %.2fM (Trainable %.2fM)" %
              (total_params, trainable_params))

    def copy_model(self, device):
        prev_net = UNetGood(n_channels=self.expert1.no_maps_as_nn_input(),
                            n_classes=self.expert2.no_maps_as_nn_output(),
                            from_exp=self.expert1,
                            to_exp=self.expert2).to(device)
        self.prev_net = nn.DataParallel(prev_net)
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

    def log_to_tb(self, writer, split_tag, wtag, losses, input, output, gt):
        writer.add_images('%s_%s/Input' % (split_tag, wtag),
                          img_for_plot(input, self.expert1.identifier),
                          self.global_step)
        writer.add_images('%s_%s/Output' % (split_tag, wtag),
                          img_for_plot(output, self.expert2.identifier),
                          self.global_step)

        writer.add_images('%s_%s/GT_EXPERT' % (split_tag, wtag),
                          img_for_plot(gt, self.expert2.identifier),
                          self.global_step)

        if self.expert2.get_task_type() == BasicExpert.TASK_REGRESSION:
            losses *= 100.
            writer.add_scalar('%s_%s/L1_Loss' % (split_tag, wtag), losses[0],
                              self.global_step)
            if len(losses) > 1:
                writer.add_scalar('%s_%s/L2_Loss' % (split_tag, wtag),
                                  losses[1], self.global_step)
        else:
            writer.add_scalar("%s_%s/CrossEntropy_Loss" % (split_tag, wtag),
                              losses[0], self.global_step)

    def train_step(self, device, writer, wtag):
        self.net.train()

        train_losses = torch.zeros(len(self.training_losses))

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            domain1, domain2_gt = batch

            domain2_gt = domain2_gt.to(device=device)

            domain2_pred = self.net(
                [domain1, self.net.module.to_exp.postprocess_train])

            backward_losses = 0
            for idx_loss, loss in enumerate(self.training_losses):
                crt_loss = loss(domain2_pred, self.gt_transform(domain2_gt))
                backward_losses += crt_loss
                # TODO: is it faster to do backward on the sum vs individually?
                train_losses[idx_loss] += crt_loss.item()
            backward_losses.backward()

            # Optimizer
            self.optimizer.step()

        train_losses /= len(self.train_loader)

        self.log_to_tb(writer, "Train", wtag, train_losses, domain1[:3],
                       domain2_pred[:3], domain2_gt[:3])

        return train_losses

    def eval_step(self, device, writer, wtag, split_tag, loader):
        self.net.eval()

        eval_losses = torch.zeros(len(self.training_losses))

        for batch in loader:
            domain1, domain2_gt = batch
            domain2_gt = domain2_gt.to(device=device)

            with torch.no_grad():
                domain2_pred = self.net(
                    [domain1, self.net.module.to_exp.postprocess_eval])

            for idx_loss, loss in enumerate(self.training_losses):
                crt_loss = loss(domain2_pred, self.gt_transform(domain2_gt))
                eval_losses[idx_loss] += crt_loss.item()

        eval_losses /= len(loader)

        self.log_to_tb(writer, split_tag, wtag, eval_losses, domain1[:3],
                       domain2_pred[:3], domain2_gt[:3])

        return eval_losses

    def train(self, start_epoch, n_epochs, device, writer, eval_test):
        self.global_step = start_epoch
        wtag = '%s_%s' % (self.expert1.identifier, self.expert2.identifier)
        epoch = 0
        for epoch in range(n_epochs):
            # 1. Train
            train_losses = self.train_step(device, writer, wtag)

            # Save model
            self.save_model(start_epoch + epoch + 1)

            # 2. Evaluate - validation set - pseudo gt from experts
            valid_losses = self.eval_step(device, writer, wtag, "Valid",
                                          self.valid_loader)

            # 3. Evaluate - test set - gt - testing on other datasets
            if eval_test and not self.test_loader == None:
                test_losses = self.eval_step(device, writer, wtag, "Test",
                                             self.test_loader)

            # 4. Scheduler
            self.scheduler.step(valid_losses.sum())

            # verbose
            if self.expert2.get_task_type() == BasicExpert.TASK_REGRESSION:
                if len(train_losses) > 1:
                    print(
                        "[%d epoch] VAL [L1_Loss %.2f   L2_Loss %.2f]   TRAIN [L1_Loss %.2f   L2_Loss %.2f]"
                        % (epoch, valid_losses[0], valid_losses[1],
                           train_losses[0], train_losses[1]))
                else:
                    print("[%d epoch] VAL [Loss %.2f]  TRAIN [Loss %.2f]" %
                          (epoch, valid_losses[0], train_losses[0]))
            else:
                print(
                    "[%d epoch] VAL [CrossEntr %.2f]   TRAIN [CrossEntr %.2f]"
                    % (epoch, valid_losses[0], train_losses[0]))

            crt_lr = self.optimizer.param_groups[0]['lr']
            print("> LR", crt_lr)
            writer.add_scalar('Train_%s/LR' % wtag, crt_lr, self.global_step)

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

                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)
                    domain2_gt = domain2_gt.to(device=device,
                                               dtype=torch.float32)

                    with torch.no_grad():
                        one_hop_pred = edge.net(
                            [domain1, edge.net.module.to_exp.postprocess_eval])

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

                    l1_per_edge[idx_edge] += 100 * edge.training_losses[0](
                        one_hop_pred, edge.gt_transform(domain2_gt)).item()
                l1_expert += 100 * edge.training_losses[0](
                    domain2_exp_gt, edge.gt_transform(domain2_gt)).item()
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

                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)

                    with torch.no_grad():
                        one_hop_pred = edge.net(
                            [domain1, edge.net.module.to_exp.postprocess_eval])

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

                    l1_per_edge[idx_edge] += 100 * edge.training_losses[0](
                        one_hop_pred,
                        edge.gt_transform(domain2_exp_gt)).item()
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
        writer.add_images(
            '%s/EXPERT' % (wtag_valid),
            img_for_plot(domain2_exp_gt[save_idxes], edge.expert2.identifier),
            0)

        # test db
        if len(test_loaders) > 0:
            writer.add_images(
                '%s/EXPERT' % (wtag_test),
                img_for_plot(domain2_exp_gt_test[save_idxes_test],
                             edge.expert2.identifier), 0)
            writer.add_images(
                '%s/GT' % (wtag_test),
                img_for_plot(domain2_gt_test[save_idxes_test],
                             edge.expert2.identifier), 0)

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
    def val_test_stats(config, writer, edges_1hop, l1_ens_valid, l1_ens_test,
                       l1_per_edge_valid, l1_per_edge_test, l1_expert_test,
                       wtag_valid, wtag_test):
        tag = "to_%s" % (edges_1hop[0].expert2.identifier)

        print("load_path", config.get('Edge Models', 'load_path'))
        print("Ensemble - sim fct: ", config.get('Ensemble', 'similarity_fct'))
        print(
            "%24s  L1(ensemble_with_expert, Expert)_valset  L1(ensemble_with_expert, GT)_testset   L1(expert, GT)_testset"
            % (tag))

        print("Loss %19s: %30.2f   %30.2f %20.2f" %
              ("Ensemble1Hop", l1_ens_valid, l1_ens_test, l1_expert_test))
        print(
            "%25s-------------------------------------------------------------------------------------"
            % (" "))

        # Show Individual Losses
        mean_l1_per_edge = np.mean(l1_per_edge_valid)
        mean_l1_per_edge_test = 0
        if len(l1_per_edge_test) > 0:
            mean_l1_per_edge_test = np.mean(l1_per_edge_test)

        idx_test_edge = 0
        for idx_edge, edge in enumerate(edges_1hop):
            writer.add_scalar(
                '1hop_%s/L1_Loss_%s' % (wtag_valid, edge.expert1.identifier),
                l1_per_edge_valid[idx_edge], 0)
            if edge.test_loader != None:
                writer.add_scalar(
                    '1hop_%s/L1_Loss_%s' %
                    (wtag_test, edge.expert1.identifier),
                    l1_per_edge_test[idx_test_edge], 0)
                print("Loss %19s: %30.2f   %30.2f" %
                      (edge.expert1.identifier, l1_per_edge_valid[idx_edge],
                       l1_per_edge_test[idx_test_edge]))
                idx_test_edge = idx_test_edge + 1
            else:
                print("Loss %19s: %30.2f    %30s" %
                      (edge.expert1.identifier, l1_per_edge_valid[idx_edge],
                       '-'))
        print(
            "%25s-------------------------------------------------------------------------------------"
            % (" "))
        print("Loss %-20s %30.2f   %30.2f" %
              ("average", mean_l1_per_edge, mean_l1_per_edge_test))

        print("")
        print("")

    def eval_1hop_ensemble_test_set(edges_1hop, device, writer, wtag):
        loaders = []
        test_edges = []
        l1_edge = []
        l1_ensemble1hop = 0
        l1_expert = 0
        save_idxes = None
        for edge in edges_1hop:
            if edge.test_loader != None:
                loaders.append(iter(edge.test_loader))
                test_edges.append(edge)
                l1_edge.append(0)

        if len(l1_edge) == 0:
            return l1_edge, l1_ensemble1hop, l1_expert, None, None, None, None

        with torch.no_grad():
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):
                domain2_1hop_ens_list = []

                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt, domain2_gt = next(loader)

                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)
                    domain2_gt = domain2_gt.to(device=device,
                                               dtype=torch.float32)

                    # Ensemble1Hop: 1hop preds
                    one_hop_pred = edge.net(
                        [domain1, edge.net.module.to_exp.postprocess_eval])
                    domain2_1hop_ens_list.append(one_hop_pred.clone())

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=False)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/output_%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(one_hop_pred[save_idxes],
                                         edge.expert2.identifier), 0)
                        writer.add_images(
                            '%s/input_%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(domain1[save_idxes],
                                         edge.expert1.identifier), 0)

                    l1_edge[idx_edge] += edge.eval_losses[0](
                        one_hop_pred, edge.gt_transform(domain2_gt)).item()

                domain2_1hop_ens_list.append(
                    edge.to_ens_transform(edge.gt_transform(domain2_exp_gt),
                                          edge.expert2.no_maps_as_ens_input()))
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                domain2_1hop_ens = edge.ensemble_filter(
                    domain2_1hop_ens_list.permute(1, 2, 3, 4, 0))

                l1_expert += edge.eval_losses[0](domain2_exp_gt,
                                                 domain2_gt).item()
                l1_ensemble1hop += edge.eval_losses[0](
                    domain2_1hop_ens, edge.gt_transform(domain2_gt)).item()

        multiply = 1.
        if edges_1hop[0].expert2.get_task_type(
        ) == BasicExpert.TASK_REGRESSION:
            multiply = 100.

        l1_edge = multiply * np.array(l1_edge) / num_batches
        l1_ensemble1hop = multiply * np.array(l1_ensemble1hop) / num_batches
        l1_expert = multiply * np.array(l1_expert) / num_batches

        return l1_edge, l1_ensemble1hop, l1_expert, domain2_1hop_ens, domain2_exp_gt, domain2_gt, save_idxes

    def eval_1hop_ensemble_valid_set(edges_1hop, device, writer, wtag):

        save_idxes = None
        loaders = []
        l1_edge = []
        l1_ensemble1hop = 0
        for edge in edges_1hop:
            loaders.append(iter(edge.valid_loader))
            l1_edge.append(0)

        with torch.no_grad():
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):
                domain2_1hop_ens_list = []
                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_exp_gt = next(loader)

                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)

                    # Ensemble1Hop: 1hop preds
                    one_hop_pred = edge.net(
                        [domain1, edge.net.module.to_exp.postprocess_eval])
                    domain2_1hop_ens_list.append(one_hop_pred.clone())

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=True)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/output_%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(one_hop_pred[save_idxes],
                                         edge.expert2.identifier), 0)
                        writer.add_images(
                            '%s/input_%s' % (wtag, edge.expert1.identifier),
                            img_for_plot(domain1[save_idxes],
                                         edge.expert1.identifier), 0)
                    l1_edge[idx_edge] += edge.eval_losses[0](
                        one_hop_pred,
                        edge.gt_transform(domain2_exp_gt)).item()

                # with_expert
                domain2_1hop_ens_list.append(
                    edge.to_ens_transform(edge.gt_transform(domain2_exp_gt),
                                          edge.expert2.no_maps_as_ens_input()))
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                domain2_1hop_ens = edge.ensemble_filter(
                    domain2_1hop_ens_list.permute(1, 2, 3, 4, 0))

                l1_ensemble1hop += edge.eval_losses[0](
                    domain2_1hop_ens,
                    edge.gt_transform(domain2_exp_gt)).item()

        multiply = 1.
        if edges_1hop[0].expert2.get_task_type(
        ) == BasicExpert.TASK_REGRESSION:
            multiply = 100.

        l1_edge = multiply * np.array(l1_edge) / num_batches
        l1_ensemble1hop = multiply * np.array(l1_ensemble1hop) / num_batches

        return l1_edge, l1_ensemble1hop, domain2_1hop_ens, domain2_exp_gt, save_idxes

    def eval_all_1hop_ensembles(edges_1hop, device, writer, config):
        if len(edges_1hop) == 0:
            return

        # === VALID ====
        wtag_valid = "to_%s_valid_set" % (edges_1hop[0].expert2.identifier)

        l1_edge_valid, l1_ens_valid, domain2_1hop_ens, domain2_gt, save_idxes_valid = Edge.eval_1hop_ensemble_valid_set(
            edges_1hop, device, writer, wtag_valid)

        # Log Valid in Tensorboard
        writer.add_images(
            '%s/output_ENSEMBLE' % (wtag_valid),
            img_for_plot(domain2_1hop_ens[save_idxes_valid],
                         edges_1hop[0].expert2.identifier), 0)
        writer.add_images(
            '%s/output_EXPERT' % (wtag_valid),
            img_for_plot(domain2_gt[save_idxes_valid],
                         edges_1hop[0].expert2.identifier), 0)

        writer.add_scalar('1hop_%s/L1_Loss_ensemble' % (wtag_valid),
                          l1_ens_valid, 0)

        del domain2_1hop_ens, domain2_gt
        torch.cuda.empty_cache()

        # === TEST ====
        wtag_test = "to_%s_test_set" % (edges_1hop[0].expert2.identifier)

        l1_edge_test, l1_ens_test, l1_expert_test, domain2_1hop_ens_test, domain2_exp_gt_test, domain2_gt_test, save_idxes_test = Edge.eval_1hop_ensemble_test_set(
            edges_1hop, device, writer, wtag_test)

        if len(l1_edge_test) > 0:
            # # Log Test in Tensorboard
            writer.add_images(
                '%s/output_ENSEMBLE' % (wtag_test),
                img_for_plot(domain2_1hop_ens_test[save_idxes_test],
                             edges_1hop[0].expert2.identifier), 0)
            writer.add_images(
                '%s/output_EXPERT' % (wtag_test),
                img_for_plot(domain2_exp_gt_test[save_idxes_test],
                             edges_1hop[0].expert2.identifier), 0)

            writer.add_images(
                '%s/output_GT' % (wtag_test),
                img_for_plot(domain2_gt_test[save_idxes_test],
                             edges_1hop[0].expert2.identifier), 0)
            writer.add_scalar('1hop_%s/L1_Loss_ensemble' % (wtag_test),
                              l1_ens_test, 0)

        # Val+Test STATS
        Edge.val_test_stats(config, writer, edges_1hop, l1_ens_valid,
                            l1_ens_test, l1_edge_valid, l1_edge_test,
                            l1_expert_test, wtag_valid, wtag_test)

        return

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

                    domain2_exp_gt = domain2_exp_gt.to(device=device,
                                                       dtype=torch.float32)

                    with torch.no_grad():
                        # Ensemble1Hop: 1hop preds
                        one_hop_pred = edge.net(
                            [domain1, edge.net.module.to_exp.postprocess_eval])
                        domain2_1hop_ens_list.append(one_hop_pred.clone())

                # with_expert
                domain2_1hop_ens_list.append(domain2_exp_gt)
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                domain2_1hop_ens = edge.ensemble_filter(
                    domain2_1hop_ens_list.permute(1, 2, 3, 4, 0))

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
