import argparse
from datetime import datetime
import glob
import os
import subprocess
import sys
import pprint
import logging as log
import multiprocessing

import matplotlib.pyplot
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from lib.datasets import *
from lib.diffutils import positional_encoding, gradient
from lib.models import *
from lib.renderer import Renderer
from lib.tracer import *
from lib.utils import PerfTimer, image_to_np, suppress_output
from lib.validator import *

class Trainer(object):
    """
    Base class for the trainer:

    init()
    |- set_dataset()
    |- set_network()
    |- set_optimizer()
    |- set_renderer()
    |- set_logger()

    train():
        for every epoch:
            pre_epoch()
                resample()

            iterate()
                step()

            post_epoch()
            |- log_tb()
            |- render_tb()
            |- save_model()

            validate()
    """

    #######################
    # __init__
    #######################
    
    def __init__(self, args, args_str):
        """Constructor.
        
        Args:
            args (Namespace): parameters
            args_str (str): string representation of all parameters
            model_name (str): model nametag
        """
        #torch.multiprocessing.set_start_method('spawn')
        multiprocessing.set_start_method('spawn')

        self.args = args 
        self.args_str = args_str
        self.args.epochs += 1
        self.timer = PerfTimer(activate=self.args.perf)
        self.timer.reset()
        
        # Set device to use
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        # In-training variables
        self.latents = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.dataset_size = None
        self.log_dict = {}

        # Initialize
        self.set_dataset()
        self.timer.check('set_dataset')
        self.set_network()
        self.timer.check('set_network')
        self.set_optimizer()
        self.timer.check('set_optimizer')
        self.set_renderer()
        self.timer.check('set_renderer')
        self.set_logger()
        self.timer.check('set_logger')
        self.set_validator()
        self.timer.check('set_validator')
        
    #######################
    # set_ functions
    #######################

    def set_dataset(self):
        # 调用函数 MeshDataset(self.args)
        self.train_dataset = globals()[self.args.mesh_dataset](self.args)
        log.info("Dataset Size: {}".format(len(self.train_dataset)))
        # dataset_size 500k; batch_size 512
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=0)
        self.timer.check('create_dataloader')
        log.info("Loaded mesh dataset")
            
    def set_network(self):
        # 调用函数 OctreeSDF(self.args)
        self.net = globals()[self.args.net](self.args)
        # false
        if self.args.jit:
            self.net = torch.jit.script(self.net)
        # None
        if self.args.pretrained:
            self.net.load_state_dict(torch.load(self.args.pretrained))
        self.net.to(self.device)
        log.info("Total number of parameters: {}".format(sum(p.numel() for p in self.net.parameters())))

    def set_optimizer(self):
        # adam
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.8)
        else:
            raise ValueError('Invalid optimizer.')

    def set_renderer(self):
        # 调用函数 SphereTracer(self.args)
        self.log_tracer = globals()[self.args.tracer](self.args)
        self.renderer = Renderer(self.log_tracer, args=self.args)

    def set_logger(self):
        # armadillo
        if self.args.exp_name:
            self.log_fname = self.args.exp_name
        else:
            self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        # _results/logs/runs/armadillo
        self.log_dir = os.path.join(self.args.logs, self.log_fname)
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Parameters', self.args_str)
        log.info('Model configured and ready to go')

    def set_validator(self):
        # None
        if self.args.validator is not None:
            self.validator = globals()[self.args.validator](self.args, self.device, self.net)

    #######################
    # pre_epoch
    #######################

    def pre_epoch(self, epoch):
        # [0,1,2,3,4]
        self.loss_lods = list(range(0, self.args.num_lods))
        # -1
        if self.args.grow_every > 0:
            self.grow(epoch)
        # False
        if self.args.only_last:
            self.loss_lods = self.loss_lods[-1:]
        # 10
        if epoch % self.args.resample_every == 0:
            # 更新 self.train_dataset
            self.resample(epoch)
            log.info("Reset DataLoader")
            # 更新 self.data_loader
            self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, 
                                                    shuffle=True, pin_memory=True, num_workers=0)
            self.timer.check('create_dataloader')
        # -1; epoch start from 0
        if epoch == self.args.freeze:
            log.info('Freezing network...')
            log.info("Total number of parameters: {}".format(sum(p.numel() for p in self.net.parameters())))
            self.net.freeze()
            self.net.to(self.device)

        # 通知模型，即将开始训练
        self.net.train()
        
        # 初始化 log_dict
        self.log_dict['l2_loss'] = 0
        self.log_dict['total_loss'] = 0
        self.log_dict['total_iter_count'] = 0
        self.timer.check('pre_epoch done')

    def resample(self, epoch):
        self.train_dataset.resample()
    
    # def grow(self, epoch):
    #     stage = min(self.args.num_lods, (epoch // self.args.grow_every) + 1) # 1 indexed

    #     if self.args.growth_strategy == 'onebyone':
    #         self.loss_lods = [stage-1]
    #     elif self.args.growth_strategy == 'increase':
    #         self.loss_lods = list(range(0, stage))
    #     elif self.args.growth_strategy == 'shrink':
    #         self.loss_lods = list(range(0, self.args.num_lods))[stage-1:] 
    #     elif self.args.growth_strategy == 'finetocoarse':
    #         self.loss_lods = list(range(0, self.args.num_lods))[self.args.num_lods-stage:] 
    #     elif self.args.growth_strategy == 'onlylast':
    #         self.loss_lods = list(range(0, self.args.num_lods))[-1:] 
    #     else:
    #         raise NotImplementedError

    #######################
    # iterate
    #######################b

    def iterate(self, epoch):
        # n_iter 500K/512
        for n_iter, data in enumerate(self.train_data_loader):
            self.step_geometry(epoch, n_iter, data)

    def step_geometry(self, epoch, n_iter, data):
        # idx = n_iter + (epoch * self.dataset_size)
        # log_iter = (idx % 100 == 0)

        # Map to device
        pts = data[0].to(self.device)
        gts = data[1].to(self.device)
        nrm = data[2].to(self.device) if self.args.get_normals else None

        
        batch_size = pts.shape[0]  # 512
        self.net.zero_grad()
        # Calculate loss
        loss = 0
        l2_loss = 0.0
        _l2_loss = 0.0
        preds = []
        # False
        if self.args.return_lst:
            preds = self.net.sdf(pts, return_lst=self.args.return_lst)
            preds = [preds[i] for i in self.loss_lods]
        else:
            # level of detail 0,1,2,3,4
            for i, lod in enumerate(self.loss_lods):
                # 三线性插值与前向传播
                preds.append(self.net.sdf(pts, lod=lod))

        for pred in preds:
            _l2_loss = ((pred - gts)**2).sum()
            l2_loss += _l2_loss
        loss += l2_loss

        # Update logs
        self.log_dict['l2_loss'] += _l2_loss.item()
        self.log_dict['total_loss'] += loss.item()
        self.log_dict['total_iter_count'] += batch_size
        loss /= batch_size

        # Backpropagate
        loss.backward()
        self.optimizer.step()
    
    #######################
    # post_epoch
    #######################
    
    def post_epoch(self, epoch):
        self.net.eval()
        # log losses to tensorboard
        self.log_tb(epoch)
        # 1
        if epoch % self.args.save_every == 0:
            # save model
            self.save_model(epoch)
        # 1
        if epoch % self.args.render_every == 0:
            # render images to Tensorboard
            self.render_tb(epoch)
        self.timer.check('post_epoch done')

    def log_tb(self, epoch):
        # Log losses
        log_text = 'EPOCH {}/{}'.format(epoch+1, self.args.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        self.log_dict['l2_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | l2 loss: {:>.3E}'.format(self.log_dict['l2_loss'])
        self.writer.add_scalar('Loss/l2_loss', self.log_dict['l2_loss'], epoch)
        log.info(log_text)
        self.writer.add_scalar('Loss/total_loss', self.log_dict['total_loss'], epoch)

    def render_tb(self, epoch):
        self.net.eval()
        for d in range(self.args.num_lods):
            self.net.lod = d
            out = self.renderer.shade_images(self.net,
                                             f=self.args.camera_origin, 
                                             t=self.args.camera_lookat,
                                             fov=self.args.camera_fov).image().byte().numpy()
            self.writer.add_image(f'Depth/{d}', out.depth.transpose(2,0,1), epoch)
            self.writer.add_image(f'Hit/{d}', out.hit.transpose(2,0,1), epoch)
            self.writer.add_image(f'Normal/{d}', out.normal.transpose(2,0,1), epoch)
            self.writer.add_image(f'RGB/{d}', out.rgb.transpose(2,0,1), epoch)
            out_x = self.renderer.sdf_slice(self.net, dim=0)
            out_y = self.renderer.sdf_slice(self.net, dim=1)
            out_z = self.renderer.sdf_slice(self.net, dim=2)
            self.writer.add_image(f'Cross-section/X/{d}', image_to_np(out_x), epoch)
            self.writer.add_image(f'Cross-section/Y/{d}', image_to_np(out_y), epoch)
            self.writer.add_image(f'Cross-section/Z/{d}', image_to_np(out_z), epoch)
            self.net.lod = None
                
    def save_model(self, epoch):
        log_comps = self.log_fname.split('/')
        if len(log_comps) > 1:
            _path = os.path.join(self.args.model_path, *log_comps[:-1])
            if not os.path.exists(_path):
                os.makedirs(_path)

        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)
        # False
        if self.args.save_as_new:
            model_fname = os.path.join(self.args.model_path, f'{self.log_fname}-{epoch}.pth')
        else:
            model_fname = os.path.join(self.args.model_path, f'{self.log_fname}.pth')
        
        log.info(f'Saving model checkpoint to: {model_fname}')
        # False
        if self.args.save_all:
            torch.save(self.net, model_fname)
        else:
            torch.save(self.net.state_dict(), model_fname)
        # None
        if self.latents is not None:
            model_fname = os.path.join(self.args.model_path, f'{self.log_fname}_latents.pth')
            torch.save(self.latents.state_dict(), model_fname)

    #######################
    # train
    #######################
    
    def train(self):
        # default: None
        if self.args.validator is not None and self.args.valid_only:
            self.validate(0)
            return
        # range(250)
        for epoch in range(self.args.epochs):    
            self.timer.check('new epoch...')
            self.pre_epoch(epoch)

            if self.train_data_loader is not None:
                # 977 = 500k/512 
                self.dataset_size = len(self.train_data_loader)
            
            self.timer.check('iteration start')
            self.iterate(epoch)
            self.timer.check('iterations done')

            self.post_epoch(epoch)

            # default: None
            if self.args.validator is not None and epoch % self.args.valid_every == 0:
                self.validate(epoch)
                self.timer.check('validate')

        self.writer.close()
    
    #######################
    # validate
    #######################

    # def validate(self, epoch):
        
    #     val_dict = self.validator.validate(epoch, self.loss_lods)
        
    #     log_text = 'EPOCH {}/{}'.format(epoch, self.args.epochs)

    #     for k, v in val_dict.items():
    #         score_total = 0.0
    #         for lod, score in zip(self.loss_lods, v):
    #             self.writer.add_scalar(f'Validation/{k}/{lod}', score, epoch)
    #             score_total += score
    #         log_text += ' | {}: {:.2f}'.format(k, score_total / len(v))
    #     log.info(log_text)

