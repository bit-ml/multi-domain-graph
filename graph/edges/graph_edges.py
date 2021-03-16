import os
import re
import sys

import globals
import numpy as np
import torch
from experts.basic_expert import BasicExpert
from graph.edges.dataset2d import ImageLevelDataset
from graph.edges.unet.unet_model import get_unet
from termcolor import colored
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import utils
from utils.utils import (EnsembleFilter_SimpleMean,
                         EnsembleFilter_SimpleMedian, EnsembleFilter_TwdExpert,
                         SimScore_L1, SimScore_L2, SimScore_LPIPS,
                         SimScore_SSIM, SSIMLoss, VarianceScore, img_for_plot)

empty_fcn = (lambda x: x)


class Edge:
    def __init__(self, config, expert1, expert2, device, silent, valid_shuffle,
                 iter_no, bs_train, bs_test):
        super(Edge, self).__init__()
        self.config = config
        self.silent = silent

        self.init_logs_csv(config)
        # Analysis data
        silent_analysis = config.getboolean('Analysis', 'silent')
        if silent_analysis:
            self.log_variance_fct = lambda *args: True
            self.log_errors_fct = lambda *args: True
            self.log_metrics_fct = lambda *args: True
            self.var_score = lambda x: x
            self.logs_path = ''
            self.log_ensemble_errors_fct = lambda *args: True
            self.log_ensemble_errors_fct_complete = lambda *args: True
        else:
            # prepare output path
            self.logs_path = '%s/%s_%s/%s' % (config.get(
                'Analysis', 'LOGS_PATH'), config.get(
                    'Analysis', 'prefix'), config.get(
                        'Run id', 'datetime'), expert2.identifier)
            os.makedirs(self.logs_path, exist_ok=True)
            # get var function
            self.var_score = VarianceScore()
            self.log_variance_fct = self.log_variance

            self.log_errors_fct = lambda *args: True
            self.log_metrics_fct = lambda *args: True
            self.log_ensemble_errors_fct = lambda *args: True
            self.log_ensemble_errors_fct_complete = lambda *args: True
            '''
            self.log_errors_fct = self.log_errors
            self.log_metrics_fct = self.log_metrics
            self.analysis_score_names = ['l1', 'l2', 'ssim', 'lpips']
            self.analysis_score_fcts = [
                SimScore_L1().to(device),
                SimScore_L2().to(device),
                SimScore_SSIM(expert2.no_maps_as_nn_output(), 11).to(device),
                SimScore_LPIPS(expert2.no_maps_as_nn_output()).to(device)
            ]
            for idx in range(len(self.analysis_score_fcts)):
                self.analysis_score_fcts[idx] = nn.DataParallel(
                    self.analysis_score_fcts[idx])
            self.log_ensemble_errors_fct = self.log_ensemble_errors
            self.log_ensemble_errors_fct_complete = self.log_ensemble_errors_complete
            '''

        # Initialize ensemble model for destination task
        enable_simple_mean = config.getboolean('Ensemble',
                                               'enable_simple_mean')

        enable_simple_median = config.getboolean('Ensemble',
                                                 'enable_simple_median')
        if enable_simple_mean:
            self.ensemble_filter = EnsembleFilter_SimpleMean()
        elif enable_simple_median:
            self.ensemble_filter = EnsembleFilter_SimpleMedian()
        else:
            similarity_fcts = re.sub('\s+', '',
                                     config.get('Ensemble',
                                                'similarity_fct')).split(',')
            kernel_fct = config.get('Ensemble', 'kernel_fct')
            meanshiftiter_thresholds = np.float32(
                config.get('Ensemble', 'meanshiftiter_thresholds').split(','))
            comb_type = config.get('Ensemble', 'comb_type')
            fix_variance = config.getboolean('Ensemble', 'fix_variance')
            variance_dismiss_threshold = config.getfloat(
                'Ensemble', 'variance_dismiss_threshold')

            dist_to = config.get('Ensemble', 'dist_to')
            self.ensemble_filter = EnsembleFilter_TwdExpert(
                n_channels=expert2.no_maps_as_ens_input(),
                similarity_fcts=similarity_fcts,
                kernel_fct=kernel_fct,
                comb_type=comb_type,
                postprocess_eval=expert2.postprocess_eval,
                thresholds=meanshiftiter_thresholds,
                fix_variance=fix_variance,
                variance_th=variance_dismiss_threshold,
                dist_to=dist_to,
                dst_domain_name=expert2.domain_name,
                analysis_logs_path=self.logs_path,
                analysis_silent=silent_analysis).to(device)
        self.ensemble_filter = nn.DataParallel(self.ensemble_filter)

        model_type = config.getint('Edge Models', 'model_type')
        self.init_edge(expert1, expert2, device, model_type)

        test1 = config.get('General', 'Steps_Iter1_test')
        test2 = config.get('General', 'Steps_Iter2_test')
        if test1 or test2:
            n_workers = 8
        else:
            n_workers = max(8, 7 * torch.cuda.device_count())

        self.init_loaders(bs_train=bs_train * torch.cuda.device_count(),
                          bs_test=bs_test * torch.cuda.device_count(),
                          n_workers=n_workers,
                          valid_shuffle=valid_shuffle,
                          iter_no=iter_no)

        # OPTIMIZATION
        learning_rate = config.getfloat('Training', 'learning_rate')
        optimizer_type = config.get('Training', 'optimizer')
        weight_decay = config.getfloat('Training', 'weight_decay')
        sch_patience = config.getfloat('Training', 'reduce_lr_patience')
        sch_factor = config.getfloat('Training', 'reduce_lr_factor')
        sch_threshold = config.getfloat('Training', 'reduce_lr_threshold')
        sch_min_lr = config.getfloat('Training', 'reduce_lr_min_lr')
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
            if optimizer_type == 'adamw': optimizer_class = optim.AdamW

            self.optimizer = optimizer_class(self.net.parameters(),
                                             lr=self.lr,
                                             weight_decay=weight_decay,
                                             amsgrad=amsgrad)
        else:
            print("Incorrect optimizer", optimizer_type)
            sys.exit(-1)

        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           patience=sch_patience,
                                           factor=sch_factor,
                                           threshold=sch_threshold,
                                           min_lr=sch_min_lr)

        # TASK TYPES
        classif_losses = re.sub('\s+', '',
                                config.get('Edge Models',
                                           'classif_losses')).split(',')
        classif_losses_weights = np.float32(
            config.get('Edge Models', 'classif_losses_weights').split(','))

        regress_losses = re.sub('\s+', '',
                                config.get('Edge Models',
                                           'regression_losses')).split(',')
        regression_losses_weights = np.float32(
            config.get('Edge Models', 'regression_losses_weights').split(','))
        smoothl1_beta = np.float32(config.get('Edge Models', 'smoothl1_beta'))
        ssim_kernel = np.int32(config.get('Edge Models', 'ssim_loss_kernel'))
        loss_params = {
            "smoothl1_beta": smoothl1_beta,
            "ssim_kernel": ssim_kernel
        }

        if self.expert2.get_task_type() == BasicExpert.TASK_CLASSIFICATION:
            self.training_losses_weights = classif_losses_weights
            train_losses_str = classif_losses
            self.eval_loss = nn.CrossEntropyLoss(self.expert2.classes_weights,
                                                 reduction="none")
        else:
            self.training_losses_weights = regression_losses_weights
            train_losses_str = regress_losses
            self.eval_loss = nn.L1Loss(reduction="none")

        self.training_losses = []
        for loss_str in train_losses_str:
            loss = self.loss_from_str(loss_str, loss_params)
            self.training_losses.append(loss)

        self.gt_train_transform = self.expert2.gt_train_transform
        self.gt_eval_transform = self.expert2.gt_eval_transform
        self.gt_to_inp_transform = self.expert2.gt_to_inp_transform
        self.test_gt = self.expert2.test_gt

        self.global_step = 0
        self.trained = False

        self.test_gt = expert2.test_gt

        # CHECKPOINTing
        self.load_model_dir = os.path.join(
            config.get('Edge Models', 'load_path_iter%d' % iter_no),
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

    def init_logs_csv(self, config):
        self.logs_csv_path = config.get('Logs', 'csv_results')
        os.makedirs(self.logs_csv_path, exist_ok=True)
        self.logs_csv_path = '%s/%s_%s.csv' % (
            self.logs_csv_path, config.get('Logs', 'tensorboard_prefix'),
            config.get('Run id', 'datetime'))
        f = open(self.logs_csv_path, 'w')
        f.write(
            'enable_simple_mean, enable_simple_median, sim_fct, kernel_fct, meanshift_threshold, comb_type, fix_variance, var_th,'
        )
        f.write(
            'exp_error, direct_edge, valid_twd_exp_error, test_twd_gt_error,')
        f.write('\n')

        f.close()

    def init_edge(self, expert1, expert2, device, model_type):
        self.expert1 = expert1
        self.expert2 = expert2
        self.name = "%s -> %s" % (expert1.identifier, expert2.identifier)

        net = get_unet(model_type=model_type,
                       n_channels=expert1.no_maps_as_nn_input(),
                       n_classes=expert2.no_maps_as_nn_output(),
                       to_exp=expert2).to(device)
        self.net = nn.DataParallel(net)

        total_params = sum(p.numel() for p in self.net.parameters()) / 1e+6
        trainable_params = sum(p.numel() for p in self.net.parameters()) / 1e+6
        print("[Params %.2fM]" % (trainable_params), end=" ")

    def init_loaders(self, bs_train, bs_test, n_workers, valid_shuffle,
                     iter_no):
        # Load train dataset
        train_ds = ImageLevelDataset(self.expert1, self.expert2, self.config,
                                     iter_no, 'TRAIN')
        print("\tTrain ds", len(train_ds), end=" ")
        self.train_loader = DataLoader(train_ds,
                                       batch_size=bs_train,
                                       shuffle=True,
                                       num_workers=n_workers)

        # Load valid dataset
        valid_ds = ImageLevelDataset(self.expert1, self.expert2, self.config,
                                     iter_no, 'VALID')
        print("\tValid ds", len(valid_ds), end=" ")
        self.valid_loader = DataLoader(valid_ds,
                                       batch_size=bs_test,
                                       shuffle=valid_shuffle,
                                       num_workers=n_workers)
        # Load test dataset
        test_ds = ImageLevelDataset(self.expert1, self.expert2, self.config,
                                    iter_no, 'TEST')
        print("\tTest ds", len(test_ds))
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
                      len(next_iter_test_ds))
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

    def loss_from_str(self, loss_str, params):
        if loss_str == 'l1':
            loss = nn.L1Loss()
        elif loss_str == 'l2':
            loss = nn.MSELoss()
        elif loss_str == 'smoothl1':
            smoothl1_beta = params["smoothl1_beta"]
            loss = nn.SmoothL1Loss(beta=smoothl1_beta)
        elif loss_str == 'ssim':
            ssim_kernel = params["ssim_kernel"]
            loss = SSIMLoss(self.expert2.no_maps_as_nn_output(), ssim_kernel)
        elif loss_str == 'cross_entropy':
            loss = nn.CrossEntropyLoss(self.expert2.classes_weights)
        else:
            assert False, "Loss not defined %s" % (loss_str)
        return loss

    def __str__(self):
        return '[%s To: %s]' % (self.expert1.identifier,
                                self.expert2.identifier)

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

            domain2_pred = self.net([domain1, empty_fcn])

            backward_losses = 0
            for idx_loss, loss in enumerate(self.training_losses):
                crt_loss = loss(domain2_pred,
                                self.gt_train_transform(domain2_gt))
                backward_losses += crt_loss * self.training_losses_weights[
                    idx_loss]
                # TODO: is it faster to do backward on the sum vs individually?
                train_losses[idx_loss] += crt_loss.item()
            backward_losses.backward()

            # Optimizer
            self.optimizer.step()

        train_losses /= len(self.train_loader)

        self.log_to_tb(
            writer, "Train", wtag, train_losses, domain1[:3],
            self.net.module.to_exp.postprocess_eval(domain2_pred[:3]),
            domain2_gt[:3])

        return train_losses

    def eval_step(self, device, writer, wtag, split_tag, loader):
        self.net.eval()

        eval_losses = torch.zeros(len(self.training_losses))

        for batch in loader:
            domain1, domain2_gt = batch
            domain2_gt = domain2_gt.to(device=device)

            with torch.no_grad():
                domain2_pred = self.net([domain1, empty_fcn])

            for idx_loss, loss in enumerate(self.training_losses):
                crt_loss = loss(domain2_pred,
                                self.gt_train_transform(domain2_gt))
                eval_losses[idx_loss] += crt_loss.item()

        eval_losses /= len(loader)

        self.log_to_tb(
            writer, split_tag, wtag, eval_losses, domain1[:3],
            self.net.module.to_exp.postprocess_eval(domain2_pred[:3]),
            domain2_gt[:3])

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
            print("[%d epoch] VAL: " % epoch, end="")
            for idx in range(len(valid_losses)):
                print("Loss %.2f" % valid_losses[idx], end="  ")
            print("   TRAIN: ", end="")
            for idx in range(len(train_losses)):
                print("Loss %.2f" % train_losses[idx], end="  ")

            crt_lr = self.optimizer.param_groups[0]['lr']
            print("[LR %f]" % crt_lr)
            writer.add_scalar('Train_%s/LR' % wtag, crt_lr, self.global_step)

            self.global_step += 1

        # Save last epoch
        self.save_model(start_epoch + epoch + 1)

    ################ [Analysis logs] ###################
    def log_variance(self, ens_data, var_score_fct, logs_path, split):
        '''
            ens_data - ensemble data - containing the exp data 
            var_score_fct - fct to compute variance 
            logs_path - where we store data 
            split - indicating the data subset - 'valid' / 'test'
        '''
        nexps, bs, nchs, h, w = ens_data.shape
        file_path = os.path.join(logs_path, 'variance_%s.csv' % split)

        if os.path.exists(file_path):
            f = open(file_path, 'a')
        else:
            f = open(file_path, 'w')
            f.write('channel,variance_with_exp,variance_without_exp,\n')

        var_with_exp = var_score_fct(ens_data)
        var_without_exp = var_score_fct(ens_data[0:-1, :, :, :, :])

        var_with_exp = var_with_exp.cpu().numpy()
        var_without_exp = var_without_exp.cpu().numpy()
        for ch in range(nchs):
            ch_var_with_exp = var_with_exp[:, ch, :, :].flatten()
            ch_var_without_exp = var_without_exp[:, ch, :, :].flatten()
            for i in range(ch_var_with_exp.size):
                f.write('%d, %20.10f, %20.10f,\n' %
                        (ch, ch_var_with_exp[i], ch_var_without_exp[i]))
        f.close()

    def log_ensemble_errors(self, losses, logs_path, split, gt_type,
                            src_identifier):
        '''
            losses - computed errors
            logs_path - where we store data 
        '''
        bs, nchs, h, w = losses.shape
        file_path = os.path.join(
            logs_path,
            'ensemble_errors_%s_%s_%s.csv' % (split, gt_type, src_identifier))

        if os.path.exists(file_path):
            f = open(file_path, 'a')
        else:
            f = open(file_path, 'w')
            f.write('channel,errors,\n')

        err = losses.clone()
        err = err.cpu().numpy()
        for ch in range(nchs):
            ch_err = err[:, ch, :, :].flatten()
            for i in range(ch_err.size):
                f.write('%d, %20.10f,\n' % (ch, ch_err[i]))

        f.close()

    def log_ensemble_errors_complete(self, loss_fct, preds, target, logs_path,
                                     split, gt_type, src_identifier):
        losses = loss_fct(preds, target)
        bs, nchs, h, w = losses.shape
        file_path = os.path.join(
            logs_path,
            'ensemble_errors_%s_%s_%s.csv' % (split, gt_type, src_identifier))

        if os.path.exists(file_path):
            f = open(file_path, 'a')
        else:
            f = open(file_path, 'w')
            f.write('channel,errors,\n')

        err = losses.clone()
        err = err.cpu().numpy()
        for ch in range(nchs):
            ch_err = err[:, ch, :, :].flatten()
            for i in range(ch_err.size):
                f.write('%d, %20.10f,\n' % (ch, ch_err[i]))

        f.close()

    def log_errors(self, losses, logs_path, split, gt_type, src_identifier):
        '''
            losses - computed errors
            logs_path - where we store data 
        '''
        bs, nchs, h, w = losses.shape
        file_path = os.path.join(
            logs_path,
            'errors_%s_%s_%s.csv' % (split, gt_type, src_identifier))

        if os.path.exists(file_path):
            f = open(file_path, 'a')
        else:
            f = open(file_path, 'w')
            f.write('channel,errors,\n')

        err = losses.clone()
        err = err.cpu().numpy()
        for ch in range(nchs):
            ch_err = err[:, ch, :, :].flatten()
            for i in range(ch_err.size):
                f.write('%d, %20.10f,\n' % (ch, ch_err[i]))

        f.close()

    def log_metrics(self, res, target, logs_path, split, gt_type,
                    src_identifier):

        for idx in range(len(self.analysis_score_fcts)):

            score_name = self.analysis_score_names[idx]
            score_fct = self.analysis_score_fcts[idx]
            err = score_fct(res, target)

            bs, nchs, h, w = err.shape
            file_path = os.path.join(
                logs_path, 'score_%s_%s_%s_%s.csv' %
                (score_name, split, gt_type, src_identifier))

            if os.path.exists(file_path):
                f = open(file_path, 'a')
            else:
                f = open(file_path, 'w')
                f.write('channel,errors,\n')

            err = err.cpu().numpy()
            for ch in range(nchs):
                ch_err = err[:, ch, :, :].flatten()
                for i in range(ch_err.size):
                    f.write('%d, %20.10f,\n' % (ch, ch_err[i]))
            f.close()

    ################ [1Hop Ensembles] ##################
    def val_test_stats(config, writer, edges_1hop, l1_ens_valid, l1_ens_test,
                       l1_per_edge_valid, l1_per_edge_test, l1_expert_test,
                       wtag_valid, wtag_test):
        tag = "to_%21s" % (edges_1hop[0].expert2.identifier)

        print("Epochs", colored(config.get('Edge Models', 'start_epoch'),
                                'red'))
        print("Load Path iter1",
              colored(config.get('Edge Models', 'load_path_iter1'), 'red'))
        print("Load Path iter2",
              colored(config.get('Edge Models', 'load_path_iter2'), 'red'))

        enable_simple_mean = config.getboolean('Ensemble',
                                               'enable_simple_mean')
        enable_simple_median = config.getboolean('Ensemble',
                                                 'enable_simple_median')
        if enable_simple_mean:
            print(colored("Simple mean ensemble", 'red'))
        elif enable_simple_median:
            print(colored("Simple median ensemble", 'red'))
        else:
            print("Ensemble: ",
                  colored(config.get('Ensemble', 'similarity_fct'), 'red'))
            print("Kernel function: ",
                  colored(config.get('Ensemble', 'kernel_fct'), 'red'))
            print(
                "Meanshift Thresholds: ",
                colored(config.get('Ensemble', 'meanshiftiter_thresholds'),
                        'red'))
            print("Forward type: ",
                  colored(config.get('Ensemble', 'comb_type'), 'red'))

            print("Distance_to: ",
                  colored(config.get('Ensemble', 'dist_to'), 'red'))

            fix_variance = config.getboolean('Ensemble', 'fix_variance')
            print("Use Variance?: ", colored(str(fix_variance), 'red'))
            if fix_variance:
                print(
                    "Variance dismiss THRESHOLD: ",
                    colored(
                        config.getfloat('Ensemble',
                                        'variance_dismiss_threshold'), 'red'))
        print(
            colored(tag, "green"),
            "L1(ensemble_with_expert, Expert)_valset  L1(ensemble_with_expert, GT)_testset   L1(expert, GT)_testset"
        )

        print("Loss %19s: %30.2f   " % ("Ensemble1Hop", l1_ens_valid),
              colored("%30.3f " % l1_ens_test, 'green'),
              colored("%20.3f" % l1_expert_test, "magenta"))
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

        f = open(edges_1hop[0].logs_csv_path, 'a')
        f.write('%s,' % config.get('Ensemble', 'enable_simple_mean'))
        f.write('%s,' % config.get('Ensemble', 'enable_simple_median'))
        f.write('%s,' %
                config.get('Ensemble', 'similarity_fct').replace(',', '+'))
        f.write('%s,' % config.get('Ensemble', 'kernel_fct'))
        f.write('%s,' % config.get(
            'Ensemble', 'meanshiftiter_thresholds').replace(',', '+'))
        f.write('%s,' % config.get('Ensemble', 'comb_type'))
        f.write('%s,' % config.get('Ensemble', 'fix_variance'))
        f.write('%s,' % config.get('Ensemble', 'variance_dismiss_threshold'))
        f.write('%20.10f,' % l1_expert_test)
        # will be invalid for dst rgb
        f.write('%20.10f,' % l1_per_edge_valid[0])
        f.write('%20.10f,' % l1_ens_valid)
        f.write('%20.10f,' % l1_ens_test)
        f.write('\n')
        f.close()

    def eval_1hop_ensemble_test_set(edges_1hop, device, writer, wtag, config):
        globals.set_split('test')
        loaders = []
        test_edges = []
        l1_edge = []
        l1_edge_exp = []
        l1_ensemble1hop = []
        l1_expert = []
        save_idxes = None
        for edge in edges_1hop:
            if edge.test_loader != None:
                loaders.append(iter(edge.test_loader))
                test_edges.append(edge)
                l1_edge.append([])
                l1_edge_exp.append([])

        if len(l1_edge) == 0:
            return l1_edge, 0, 0, None, None, None, None

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

                    edge.log_metrics_fct(one_hop_pred,
                                         edge.gt_eval_transform(domain2_gt),
                                         edge.logs_path, 'test', 'gt',
                                         edge.expert1.identifier)
                    edge.log_metrics_fct(
                        one_hop_pred, edge.gt_eval_transform(domain2_exp_gt),
                        edge.logs_path, 'test', 'exp', edge.expert1.identifier)

                    crt_loss = edge.test_gt(edge.eval_loss, one_hop_pred,
                                            edge.gt_eval_transform(domain2_gt))
                    edge.log_errors_fct(crt_loss, edge.logs_path, 'test', 'gt',
                                        edge.expert1.identifier)

                    l1_edge[idx_edge] += crt_loss.view(
                        crt_loss.shape[0],
                        -1).mean(dim=1).data.cpu().numpy().tolist()

                    crt_loss = edge.eval_loss(
                        one_hop_pred, edge.gt_eval_transform(domain2_exp_gt))
                    edge.log_errors_fct(crt_loss, edge.logs_path, 'test',
                                        'exp', edge.expert1.identifier)

                    l1_edge_exp[idx_edge] += crt_loss.view(
                        crt_loss.shape[0],
                        -1).mean(dim=1).data.cpu().numpy().tolist()

                # with expert
                domain2_1hop_ens_list.append(
                    edge.gt_to_inp_transform(
                        domain2_exp_gt, edge.expert2.no_maps_as_ens_input()))
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                domain2_1hop_ens_list_perm = domain2_1hop_ens_list.permute(
                    1, 2, 3, 4, 0)

                domain2_1hop_ens = edge.ensemble_filter(
                    domain2_1hop_ens_list_perm)

                edge.log_variance_fct(domain2_1hop_ens_list, edge.var_score,
                                      edge.logs_path, 'test')

                crt_loss = edge.test_gt(
                    edge.eval_loss,
                    edge.gt_to_inp_transform(
                        domain2_exp_gt, edge.expert2.no_maps_as_ens_input()),
                    edge.gt_eval_transform(domain2_gt))

                l1_expert += crt_loss.view(
                    crt_loss.shape[0],
                    -1).mean(dim=1).data.cpu().numpy().tolist()

                edge.log_ensemble_errors_fct_complete(
                    edge.eval_loss, domain2_1hop_ens,
                    edge.gt_eval_transform(domain2_exp_gt), edge.logs_path,
                    'test', 'exp', edge.expert1.identifier)

                crt_loss = edge.test_gt(edge.eval_loss, domain2_1hop_ens,
                                        edge.gt_eval_transform(domain2_gt))
                edge.log_ensemble_errors_fct(crt_loss, edge.logs_path, 'test',
                                             'gt', edge.expert1.identifier)
                l1_ensemble1hop += crt_loss.view(
                    crt_loss.shape[0],
                    -1).mean(dim=1).data.cpu().numpy().tolist()

        multiply = 1.
        if edges_1hop[0].expert2.get_task_type(
        ) == BasicExpert.TASK_REGRESSION:
            multiply = 100.

        l1_edge = multiply * np.mean(l1_edge, axis=1)
        l1_ensemble1hop = multiply * np.mean(l1_ensemble1hop)
        l1_expert = multiply * np.mean(l1_expert)

        return l1_edge, l1_ensemble1hop, l1_expert, domain2_1hop_ens, domain2_exp_gt, domain2_gt, save_idxes

    def eval_1hop_ensemble_valid_set(edges_1hop, device, writer, wtag):
        globals.set_split('valid')
        save_idxes = None
        loaders = []
        l1_edge = []
        l1_ensemble1hop = []
        for edge in edges_1hop:
            loaders.append(iter(edge.valid_loader))
            l1_edge.append([])

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

                    edge.log_metrics_fct(
                        one_hop_pred, edge.gt_eval_transform(domain2_exp_gt),
                        edge.logs_path, 'valid', 'exp',
                        edge.expert1.identifier)

                    crt_loss = edge.eval_loss(
                        one_hop_pred, edge.gt_eval_transform(domain2_exp_gt))
                    edge.log_errors_fct(crt_loss, edge.logs_path, 'valid',
                                        'exp', edge.expert1.identifier)
                    l1_edge[idx_edge] += crt_loss.view(
                        crt_loss.shape[0],
                        -1).mean(dim=1).data.cpu().numpy().tolist()

                # with_expert
                domain2_1hop_ens_list.append(
                    edge.gt_to_inp_transform(
                        domain2_exp_gt, edge.expert2.no_maps_as_ens_input()))
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                domain2_1hop_ens_list_perm = domain2_1hop_ens_list.permute(
                    1, 2, 3, 4, 0)

                domain2_1hop_ens = edge.ensemble_filter(
                    domain2_1hop_ens_list_perm)

                edge.log_variance_fct(domain2_1hop_ens_list, edge.var_score,
                                      edge.logs_path, 'valid')

                crt_loss = edge.eval_loss(
                    domain2_1hop_ens, edge.gt_eval_transform(domain2_exp_gt))
                edge.log_ensemble_errors_fct(crt_loss, edge.logs_path, 'valid',
                                             'exp', edge.expert1.identifier)
                l1_ensemble1hop += crt_loss.view(
                    domain2_1hop_ens.shape[0],
                    -1).mean(dim=1).data.cpu().numpy().tolist()

        multiply = 1.
        if edges_1hop[0].expert2.get_task_type(
        ) == BasicExpert.TASK_REGRESSION:
            multiply = 100.

        l1_edge = multiply * np.mean(l1_edge, axis=1)
        l1_ensemble1hop = multiply * np.mean(l1_ensemble1hop)

        return l1_edge, l1_ensemble1hop, domain2_1hop_ens, domain2_exp_gt, save_idxes

    def eval_all_1hop_ensembles(edges_1hop, device, writer, config):
        enable_simple_mean = config.getboolean('Ensemble',
                                               'enable_simple_mean')
        enable_simple_median = config.getboolean('Ensemble',
                                                 'enable_simple_median')
        if enable_simple_mean:
            print(colored("Simple mean ensemble", 'red'))
        elif enable_simple_median:
            print(colored("Simple median ensemble", 'red'))
        else:
            print("Ensemble: ",
                  colored(config.get('Ensemble', 'similarity_fct'), 'red'))
            print("Kernel function: ",
                  colored(config.get('Ensemble', 'kernel_fct'), 'red'))
            print(
                "Meanshift Thresholds: ",
                colored(config.get('Ensemble', 'meanshiftiter_thresholds'),
                        'red'))
            print("Forward type: ",
                  colored(config.get('Ensemble', 'comb_type'), 'red'))
            print("Distance_to: ",
                  colored(config.get('Ensemble', 'dist_to'), 'red'))
            fix_variance = config.getboolean('Ensemble', 'fix_variance')
            print("Use Variance?: ", colored(str(fix_variance), 'red'))
            if fix_variance:
                print(
                    "Variance dismiss THRESHOLD: ",
                    colored(
                        config.getfloat('Ensemble',
                                        'variance_dismiss_threshold'), 'red'))

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
            edges_1hop, device, writer, wtag_test, config)

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
        edges_order = np.argsort(
            l1_edge_test)  # first - best edge according to testing set

        return edges_order

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
                domain2_1hop_ens_list.append(
                    edge.gt_to_inp_transform(
                        domain2_exp_gt, edge.expert2.no_maps_as_ens_input()))
                domain2_1hop_ens_list = torch.stack(domain2_1hop_ens_list)

                domain2_1hop_ens_list_perm = domain2_1hop_ens_list.permute(
                    1, 2, 3, 4, 0)
                domain2_1hop_ens = edge.ensemble_filter(
                    domain2_1hop_ens_list_perm)

                if edge.expert2.get_task_type(
                ) == BasicExpert.TASK_CLASSIFICATION:
                    domain2_1hop_ens = domain2_1hop_ens.argmax(
                        dim=1, keepdims=True).long()

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
