import glob
import logging
import os
import pathlib
import time
from datetime import datetime

import numpy as np
import torch
from graph.edges.dataset2d import Domain2DDataset, DomainTestDataset
from graph.edges.unet.unet_model import UNetGood
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import utils
from utils.utils import img_for_plot


class Edge:
    def __init__(self, config, expert1, expert2, device, rnd_sampler, silent,
                 valid_shuffle):
        super(Edge, self).__init__()
        self.config = config
        self.silent = silent

        self.init_edge(expert1, expert2, device)
        self.init_loaders(bs=100,
                          bs_test=220 * torch.cuda.device_count(),
                          n_workers=4,
                          rnd_sampler=rnd_sampler,
                          valid_shuffle=valid_shuffle)

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

        self.load_model_dir = os.path.join(
            config.get('Edge Models', 'load_path'),
            '%s_%s' % (expert1.str_id, expert2.str_id))

        if config.getboolean('Edge Models', 'save_models'):
            self.save_model_dir = os.path.join(
                config.get('Edge Models', 'save_path'),
                config.get('Run id', 'datetime'),
                '%s_%s' % (expert1.str_id, expert2.str_id))
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
        self.name = "%s -> %s" % (expert1.domain_name, expert2.domain_name)
        self.net = UNetGood(n_channels=self.expert1.n_maps,
                            n_classes=self.expert2.n_maps,
                            bilinear=True).to(device)
        self.net = nn.DataParallel(self.net)

    def init_loaders(self, bs, bs_test, n_workers, rnd_sampler, valid_shuffle):
        RGBS_PATH = self.config.get('Paths', 'RGBS_PATH')
        EXPERTS_OUTPUT_PATH = self.config.get('Paths', 'EXPERTS_OUTPUT_PATH')
        TRAIN_PATH = self.config.get('Paths', 'TRAIN_PATH')
        VALID_PATH = self.config.get('Paths', 'VALID_PATH')

        experts = [self.expert1, self.expert2]
        train_ds = Domain2DDataset(RGBS_PATH, EXPERTS_OUTPUT_PATH, TRAIN_PATH,
                                   experts)
        valid_ds = Domain2DDataset(RGBS_PATH, EXPERTS_OUTPUT_PATH, VALID_PATH,
                                   experts)

        self.train_loader = DataLoader(train_ds,
                                       batch_size=bs,
                                       shuffle=True,
                                       num_workers=n_workers)
        self.valid_loader = DataLoader(
            valid_ds,
            batch_size=bs,
            shuffle=valid_shuffle,
            # sampler=rnd_sampler,
            num_workers=n_workers)

        test_ds = DomainTestDataset(self.expert1.identifier,
                                    self.expert2.identifier)
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
            assert domain1.shape[1] == self.net.n_channels
            assert domain2_gt.shape[1] == self.net.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            domain2_pred = self.net(domain1)
            l2_loss = self.l2(domain2_pred, domain2_gt)
            train_l2_loss += l2_loss.item()

            with torch.no_grad():
                train_l1_loss += self.l1(domain2_pred, domain2_gt).item()
            # print("-----train_loss", train_loss, domain2_pred.shape)

            # Optimizer
            self.optimizer.zero_grad()
            l2_loss.backward()
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
            assert domain1.shape[1] == self.net.n_channels
            assert domain2_gt.shape[1] == self.net.n_classes

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

    def train(self, epochs, device, writer, eval_test):
        wtag = '%s_%s' % (self.expert1.str_id, self.expert2.str_id)
        for epoch in range(epochs):
            # 1. Train
            train_l2_loss, train_l1_loss = self.train_step(
                device, writer, wtag)
            writer.add_scalar("Train_%s/L2_Loss" % wtag, train_l2_loss,
                              self.global_step)
            writer.add_scalar('Train_%s/L1_Loss' % wtag, train_l1_loss,
                              self.global_step)

            # Save model
            self.save_model(epoch + 1)

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
            self.scheduler.step(val_l2_loss)
            writer.add_scalar('Train_%s/LR' % wtag,
                              self.optimizer.param_groups[0]['lr'],
                              self.global_step)

            self.global_step += 1

        # Save last epoch
        self.save_model(epoch + 1)

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

        correlations = np.zeros((len(edges_1hop) + 1, len(edges_1hop) + 1))
        num_batches = len(valid_loaders[0])
        n_samples = 0
        for idx_batch in range(num_batches):
            domain2_1hop_ens_list = []

            for idx_edge, data_edge in enumerate(zip(edges_1hop,
                                                     valid_loaders)):
                edge, loader = data_edge
                domain1, domain2_gt = next(loader)

                assert domain1.shape[1] == edge.net.module.n_channels
                assert domain2_gt.shape[1] == edge.net.module.n_classes

                domain1 = domain1.to(device=device, dtype=torch.float32)
                domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    one_hop_pred = edge.net(domain1)
                    domain2_1hop_ens_list.append(
                        one_hop_pred.clone().data.cpu().numpy())
            domain2_1hop_ens_list.append(domain2_gt.cpu().numpy())

            # domain2_1hop_ens_list contains all data for current batch
            correlations = utils.get_correlation_score(domain2_1hop_ens_list,
                                                       correlations,
                                                       drop_version)
            n_samples = n_samples + domain1.shape[0]
        correlations = correlations / n_samples
        return correlations

    ################ [1Hop Ensembles] ##################
    def eval_1hop_ensemble_test_set(loaders, l1_per_edge, l1_ensemble1hop,
                                    edges_1hop, device, save_idxes, writer,
                                    wtag):
        with torch.no_grad():
            num_batches = len(loaders[0])
            for idx_batch in tqdm(range(num_batches)):
                domain2_1hop_ens_list = []

                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_gt, domain2_pseudo_gt = next(loader)

                    assert domain1.shape[1] == edge.net.module.n_channels
                    assert domain2_gt.shape[1] == edge.net.module.n_classes

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_gt = domain2_gt.to(device=device,
                                               dtype=torch.float32)
                    domain2_pseudo_gt = domain2_pseudo_gt.to(
                        device=device, dtype=torch.float32)

                    with torch.no_grad():
                        # Ensemble1Hop: 1hop preds
                        one_hop_pred = edge.net(domain1)
                        domain2_1hop_ens_list = one_hop_pred.clone()

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=False)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/%s' % (wtag, edge.expert1.str_id),
                            img_for_plot(one_hop_pred[save_idxes]), 0)

                    l1_per_edge[idx_edge] += edge.l1(one_hop_pred,
                                                     domain2_gt).item()

                domain2_1hop_ens_list = torch.stack(
                    [domain2_1hop_ens_list, domain2_pseudo_gt])
                domain2_1hop_ens = utils.combine_maps(domain2_1hop_ens_list)
                l1_ensemble1hop += edge.l1(domain2_1hop_ens, domain2_gt).item()

            return l1_per_edge, l1_ensemble1hop, save_idxes, domain2_1hop_ens, domain2_gt, num_batches

    def eval_1hop_ensemble_valid_set(loaders, l1_per_edge, l1_ensemble1hop,
                                     edges_1hop, device, save_idxes, writer,
                                     wtag):
        with torch.no_grad():
            num_batches = len(loaders[0])
            for idx_batch in range(num_batches):

                for idx_edge, data_edge in enumerate(zip(edges_1hop, loaders)):
                    edge, loader = data_edge
                    domain1, domain2_gt = next(loader)

                    assert domain1.shape[1] == edge.net.module.n_channels
                    assert domain2_gt.shape[1] == edge.net.module.n_classes

                    domain1 = domain1.to(device=device, dtype=torch.float32)
                    domain2_gt = domain2_gt.to(device=device,
                                               dtype=torch.float32)

                    with torch.no_grad():
                        # Ensemble1Hop: 1hop preds
                        one_hop_pred = edge.net(domain1)
                        domain2_1hop_ens_list = one_hop_pred.clone()

                    if idx_batch == len(loader) - 1:
                        if save_idxes is None:
                            save_idxes = np.random.choice(domain1.shape[0],
                                                          size=(3),
                                                          replace=False)
                        # Show last but one batch edges
                        writer.add_images(
                            '%s/%s' % (wtag, edge.expert1.str_id),
                            img_for_plot(one_hop_pred[save_idxes]), 0)

                    l1_per_edge[idx_edge] += edge.l1(one_hop_pred,
                                                     domain2_gt).item()

                domain2_1hop_ens_list = torch.stack(
                    [domain2_1hop_ens_list, domain2_gt])
                domain2_1hop_ens = utils.combine_maps(domain2_1hop_ens_list)
                l1_ensemble1hop += edge.l1(domain2_1hop_ens, domain2_gt).item()

            return l1_per_edge, l1_ensemble1hop, save_idxes, domain2_1hop_ens, domain2_gt, num_batches

    def eval_1hop_ensemble(edges_1hop, save_idxes, save_idxes_test, device,
                           writer, with_drop):
        if with_drop == 1:
            wtag_valid = "to_%s_valid_set_with_drop" % edges_1hop[
                0].expert2.str_id
            wtag_test = "to_%s_test_set_with_drop" % edges_1hop[
                0].expert2.str_id
        else:
            wtag_valid = "to_%s_valid_set_no_drop" % edges_1hop[
                0].expert2.str_id
            wtag_test = "to_%s_test_set_no_drop" % edges_1hop[0].expert2.str_id

        valid_loaders = []
        l1_per_edge = []
        l1_ensemble1hop = 0
        for edge in edges_1hop:
            valid_loaders.append(iter(edge.valid_loader))
            l1_per_edge.append(0)

        start = time.time()
        l1_per_edge, l1_ensemble1hop, save_idxes, domain2_1hop_ens, domain2_gt, num_batches = Edge.eval_1hop_ensemble_valid_set(
            valid_loaders, l1_per_edge, l1_ensemble1hop, edges_1hop, device,
            save_idxes, writer, wtag_valid)
        end = time.time()
        print("VALID Edge.eval_1hop_ensemble_aux", end - start)

        test_loaders = []
        test_edges = []
        l1_per_edge_test = []
        l1_ensemble1hop_test = 0
        for edge in edges_1hop:
            if edge.test_loader != None:
                test_loaders.append(iter(edge.test_loader))
                test_edges.append(edge)
                l1_per_edge_test.append(0)

        start = time.time()
        if len(test_loaders) > 0:
            l1_per_edge_test, l1_ensemble1hop_test, save_idxes_test, domain2_1hop_ens_test, domain2_gt_test, num_batches_test = Edge.eval_1hop_ensemble_test_set(
                test_loaders, l1_per_edge_test, l1_ensemble1hop_test,
                test_edges, device, save_idxes_test, writer, wtag_test)
        end = time.time()
        print("TEST Edge.eval_1hop_ensemble_test_set", end - start)

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
            writer.add_images('%s/GT' % (wtag_test),
                              img_for_plot(domain2_gt_test[save_idxes_test]),
                              0)
            l1_ensemble1hop_test = np.array(
                l1_ensemble1hop_test) / num_batches_test
            writer.add_scalar('1hop_%s/L1_Loss_ensemble' % (wtag_test),
                              l1_ensemble1hop_test, 0)

        if with_drop:
            tag = "to_%s_with_drop" % (edges_1hop[0].expert2.str_id)
        else:
            tag = "to_%s_no_drop" % (edges_1hop[0].expert2.str_id)
        print("%24s  L1(ensemble, Expert)_valset  L1(ensemble, GT)_testset" %
              (tag))
        print("Loss %19s: %20.2f   %20.2f" %
              ("Ensemble1Hop", l1_ensemble1hop, l1_ensemble1hop_test))
        print("%25s-----------------------------------------------------" %
              (" "))

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
                '1hop_%s/L1_Loss_%s' % (wtag_valid, edge.expert1.str_id),
                l1_per_edge[idx_edge], 0)
            if edge.test_loader != None:
                writer.add_scalar(
                    '1hop_%s/L1_Loss_%s' % (wtag_test, edge.expert1.str_id),
                    l1_per_edge_test[idx_test_edge], 0)
                print("Loss %19s: %20.2f   %20.2f" %
                      (edge.expert1.str_id, l1_per_edge[idx_edge],
                       l1_per_edge_test[idx_test_edge]))
                idx_test_edge = idx_test_edge + 1
            else:
                print("Loss %19s: %.2f    - " %
                      (edge.expert1.str_id, l1_per_edge[idx_edge]))
        print("%25s-----------------------------------------------------" %
              (" "))
        print("Loss %-20s %20.2f   %20.2f" %
              ("average", mean_l1_per_edge, mean_l1_per_edge_test))

        print("")
        print("")
        return save_idxes, save_idxes_test

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
        tag = 'Valid_2hop_%s_%s' % (self.expert1.str_id, self.expert2.str_id)

        for idx_batch, batch in enumerate(self.valid_loader):
            domain1, domain2_expertgt = batch
            assert domain1.shape[1] == self.net.n_channels
            assert domain2_expertgt.shape[1] == self.net.n_classes

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
                    experts_names = "%s-%s-%s" % (edge1.expert1.str_id,
                                                  edge1.expert2.str_id,
                                                  edge2.expert2.str_id)

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
        wtag = '%s_%s' % (self.expert1.str_id, self.expert2.str_id)
        start_id = self.expert1.str_id
        end_id = self.expert2.str_id
        edges_2hop = []
        edges_1hop = []

        print("Direct Edge X->Y: %s" % (self))
        for edge_xk in graph.edges:
            if edge_xk.ill_posed:
                continue
            if edge_xk.expert1.str_id == start_id:
                if edge_xk.expert2.str_id == end_id:
                    continue

                for edge_ky in graph.edges:
                    if edge_ky.expert1.str_id == edge_xk.expert2.str_id and edge_ky.expert2.str_id == end_id:
                        if edge_ky.ill_posed:
                            continue
                        print("\tAdding 2hop: %s, %s" % (edge_xk, edge_ky))
                        edges_2hop.append((edge_xk, edge_ky))
            else:
                if edge_xk.expert2.str_id == end_id:
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
