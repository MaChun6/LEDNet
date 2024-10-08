from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, PerceptualLoss, WeightedTVLoss, g_path_regularize,FFTLoss, Stripformer_Loss,
                     gradient_penalty_loss, r1_penalty, PSNRLoss, FFTWeightLoss, SSIMWeightLoss, SSIMLoss, BCELoss)

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss', 'GANLoss', 'gradient_penalty_loss','Stripformer_Loss',
    'r1_penalty', 'g_path_regularize', 'FFTLoss', 'PSNRLoss', 'FFTWeightLoss', 'SSIMWeightLoss', 'SSIMLoss', 'BCELoss'
]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must constain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
