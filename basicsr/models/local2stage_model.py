import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, tensor2img_fast
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class Local2StageModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(Local2StageModel, self).__init__(opt)
        # Set random seed and deterministic
        # define network
        self.net_g = build_network(opt['network_g'])
        self.init_weights = self.opt['train'].get('init_weights', False)
        if self.init_weights:
            self.initialize_weights(self.net_g, 0.1)
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def initialize_weights(self, net_l, scale=0.1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for n, m in net.named_modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('mask_opt'):
            self.cri_mask = build_loss(train_opt['mask_opt']).to(self.device)
        else:
            self.cri_mask = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        self.use_side_loss = train_opt.get('use_side_loss', True)
        self.side_loss_weight = train_opt.get('side_loss_weight', 0.8)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
