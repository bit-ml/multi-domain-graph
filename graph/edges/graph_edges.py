import logging

import torch
from graph.edges.dataset2d import Domain2DDataset
from graph.edges.unet.unet_model import UNet
from torch import nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.utils import DummySummaryWriter

RGBS_PATH = "/data/tracking-vot/GOT-10k/"
TRAIN_RGBS_PATH = "%s/train/" % RGBS_PATH
VALID_RGBS_PATH = "%s/val" % RGBS_PATH


class Edge:
    def __init__(self, expert1, expert2, device):
        super(Edge, self).__init__()
        self.init_edge(expert1, expert2, device)
        self.init_loaders(bs=10, n_workers=0)

        self.lr = 0.001
        self.optimizer = RMSprop(self.net.parameters(),
                                 lr=self.lr,
                                 weight_decay=1e-8,
                                 momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode='max',
                                           patience=2)
        self.criterion = nn.BCEWithLogitsLoss()
        self.global_step = 0

        self.writer = DummySummaryWriter()
        # writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}', disable-)

    def init_edge(self, expert1, expert2, device):
        self.expert1 = expert1
        self.expert2 = expert2
        self.name = "%s -> %s" % (expert1.domain_name, expert2.domain_name)
        self.net = UNet(n_channels=self.expert1.n_maps,
                        n_classes=self.expert2.n_maps,
                        bilinear=True).to(device)

        logging.info(f'''Edge Initialized:
            From: {self.expert1.domain_name}
            To:   {self.expert2.domain_name}
        ''')

    def init_loaders(self, bs, n_workers):
        train_rgbs_path = TRAIN_RGBS_PATH
        valid_rgbs_path = VALID_RGBS_PATH
        experts = [self.expert1, self.expert2]
        train_ds = Domain2DDataset(train_rgbs_path, experts)
        valid_ds = Domain2DDataset(valid_rgbs_path, experts)

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

        train_loss = 0

        for batch in self.train_loader:
            domain1, domain2_gt = batch
            assert domain1.shape[1] == self.net.n_channels
            assert domain2_gt.shape[1] == self.net.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            domain2_pred = self.net(domain1)
            loss = self.criterion(domain2_pred, domain2_gt)
            train_loss += loss.item()
            print("-----train_loss", train_loss, domain2_pred.shape)
            self.writer.add_scalar('Loss/train', loss.item(), self.global_step)

            # Optimizer
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
            self.optimizer.step()

        return train_loss / len(self.train_loader)

    def eval_step(self, device):
        """Evaluation without the densecrf with the dice coefficient"""
        self.net.eval()
        eval_loss = 0

        for batch in self.valid_loader:
            domain1, domain2_gt = batch
            assert domain1.shape[1] == self.net.n_channels
            assert domain2_gt.shape[1] == self.net.n_classes

            domain1 = domain1.to(device=device, dtype=torch.float32)
            domain2_gt = domain2_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                domain2_pred = self.net(domain1)
            loss = self.criterion(domain2_pred, domain2_gt)

            eval_loss += loss.item()
            print("-----eval_loss", eval_loss, domain2_pred.shape)

        return eval_loss / len(self.valid_loader)

    def train(self, epochs, device, save_cp=True):
        for epoch in range(epochs):
            train_loss = self.train_step(device)
            val_loss = self.eval_step(device)
            print("train_loss", train_loss)
            print("val_loss", val_loss)

            self.global_step += 1

            # # Scheduler
            # self.scheduler.step(val_score)

            # if global_step % (n_train // (10 * batch_size)) == 0:
            #     for tag, value in net.named_parameters():
            #         tag = tag.replace('.', '/')
            #         writer.add_histogram('weights/' + tag,
            #                              value.data.cpu().numpy(),
            #                              global_step)
            #         writer.add_histogram('grads/' + tag,
            #                              value.grad.data.cpu().numpy(),
            #                              global_step)
            #     val_score = eval_net(net, val_loader, device)
            #     writer.add_scalar('learning_rate',
            #                       optimizer.param_groups[0]['lr'],
            #                       global_step)

            #     if net.n_classes > 1:
            #         logging.info('Validation cross entropy: {}'.format(
            #             val_score))
            #         writer.add_scalar('Loss/test', val_score,
            #                           global_step)
            #     else:
            #         logging.info(
            #             'Validation Dice Coeff: {}'.format(val_score))
            #         writer.add_scalar('Dice/test', val_score,
            #                           global_step)

            #     writer.add_images('images', imgs, global_step)
            #     if net.n_classes == 1:
            #         writer.add_images('masks/true', true_masks,
            #                           global_step)
            #         writer.add_images('masks/pred',
            #                           torch.sigmoid(masks_pred) > 0.5,
            #                           global_step)

        # if save_cp:
        #     try:
        #         os.mkdir(dir_checkpoint)
        #         logging.info('Created checkpoint directory')
        #     except OSError:
        #         pass
        #     torch.save(net.state_dict(),
        #                dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        #     logging.info(f'Checkpoint {epoch + 1} saved !')

    # writer.close()
