import random
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.use_flip = opt.get('use_flip', True)
        self.use_rot = opt.get('use_rot', True)
        self.crop_size = opt.get('crop_size', 256)
        self.scale = opt.get('scale', 1)
        self.multiple_width = opt.get('scale', 16)

        if ',' in opt['dataroot_gt']:
            gts = opt['dataroot_gt'].split(',')
            self.gt_folder = gts[:1] * 5
            self.gt_folder.extend(gts[1:])
            lqs = opt['dataroot_lq'].split(',')
            self.lq_folder = lqs[:1] * 5
            self.lq_folder.extend(lqs[1:])
        else:
            self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
            
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        if self.opt['phase'] == 'train':
            random.shuffle(self.paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        if 'relo' in gt_path:
            mask_path = gt_path.replace('image-test', 'mask').replace('/sharp', '').replace('_sharp', '')
            img_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)/255.0
            h, w = img_mask.shape[:2]
            nw, nh = round(0.5 * w), round(0.5 * h)
            img_mask = cv2.resize(img_mask, (nw, nh), interpolation=cv2.INTER_CUBIC)
            img_mask[img_mask>=0.5] = 1
            img_mask[img_mask<0.5] = 0

        elif 'lolblur' in gt_path:
            img_mask = np.ones(img_lq.shape, dtype=np.float32)

        assert os.path.basename(gt_path.replace('sharp','')) == os.path.basename(lq_path.replace('blur','')), f'name error gt: {gt_path},  lq: {lq_path}'
        assert (img_gt != img_lq).any(), f'gt and lq should not be the same, gt: {gt_path},  lq: {lq_path}'

        # augmentation for training
        if self.opt['phase'] == 'train':
            # random crop
            img_gt, img_lq, img_mask = random_mask_crop(img_gt, img_lq, img_mask, self.crop_size)
            # flip, rotation
            img_gt, img_lq, img_mask = augment([img_gt, img_lq, img_mask], self.use_flip, self.use_rot)
        else:
            if img_gt.shape[0] % self.multiple_width != 0:
                l_pad = self.multiple_width - img_gt.shape[0] % self.multiple_width
                img_lq = np.pad(img_lq, ((0, l_pad), (0, 0), (0, 0)), mode='reflect')
                # img_mask = np.pad(img_mask, ((0, l_pad), (0, 0), (0, 0)), mode='constant', constant_values=0)
            if img_gt.shape[1] % self.multiple_width != 0:
                l_pad = self.multiple_width - img_gt.shape[1] % self.multiple_width
                img_lq = np.pad(img_lq, ((0, 0), (0, l_pad), (0, 0)), mode='reflect')
                # img_mask = np.pad(img_mask, ((0, 0), (0, l_pad), (0, 0)), mode='constant', constant_values=0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_mask = img2tensor([img_gt, img_lq, img_mask], bgr2rgb=True, float32=True)
        # normalize
        normalize(img_lq, self.mean, self.std, inplace=True)
        normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'mask': img_mask[:1,:,:], 'lq_path': lq_path, 'gt_path': gt_path}

if __name__ == '__main__':
    import cv2
    import os
    import numpy as np
    opt_path = '/data///code/llblur/LEDNet-master/options/llbnetv72.yml'
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
    opt = opt['datasets']['train']
    dataset = PairedImageDataset(opt)
    print(len(dataset))
    data = dataset[0]
    print(data.keys())
