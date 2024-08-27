# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
from distutils.log import error
import os
from turtle import down
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img, scandir
from basicsr.utils.download_util import load_file_from_url
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from tqdm import tqdm

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='/data/vjuicefs_ai_camera_llm/11165663/datasets/relo/image/test/blur')
    parser.add_argument('--result_path', type=str, default='/data/vjuicefs_ai_camera_llm/11165663/code/llblur/LLBlur_test/results')
    parser.add_argument('--pretrain_path', type=str, default='/data/vjuicefs_ai_camera_llm/11165663/code/llblur/LEDNet-master/experiments/20240624_150503_LEDNet_local/models/net_g_480000.pth')
    parser.add_argument('--model', type=str, default='lednet',
                                    help='options: lednet, lednet_retrain, lednetgan')
    args = parser.parse_args()
    model_path_dic = {'lednet': '/data///code/llblur/LEDNet-master/experiments/20240624_150503_LEDNet_local/models/net_g_480000.pth'}

    # ------------------------ input & output ------------------------
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    if args.result_path.endswith('/'):  # solve when path ends with /
        args.result_path = args.result_path[:-1]
    result_root = f'{args.result_path}/{os.path.basename(args.test_path)}'
    args.result_path = args.result_path + '/' + f'{args.model}_local'

    # ------------------ set up LEDNet network -------------------
    down_factor = 8 # check_image_size
    net = ARCH_REGISTRY.get('LEDNet')(channels=[32, 64, 128, 128], connection=False).to(device)
    
    # ckpt_path = 'weights/lednet.pth'
    ckpt_path = model_path_dic[args.model]
    checkpoint = torch.load(ckpt_path)['params']
    net.load_state_dict(checkpoint)
    net.eval()

    # -------------------- start to processing ---------------------
    # scan all the jpg and png images
    img_paths = sorted(list(scandir(args.test_path, suffix=('jpg', 'png'), recursive=True, full_path=True)))

    for img_path in tqdm(img_paths):
        print(f'Processing: {img_path}')
        save_restore_path = img_path.replace(args.test_path, args.result_path)
        save_dirname = os.path.dirname(save_restore_path)
        os.makedirs(save_dirname, exist_ok=True)
        print(f'\n{img_path} saved in {save_restore_path}, {save_dirname}')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)

        # without [-1,1] normalization in lednet model (paper version) 
        normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img_t = img_t.unsqueeze(0).to(device)

        # lednet inference
        with torch.no_grad():
            # check_image_size
            H, W = img_t.shape[2:]
            img_t = check_image_size(img_t, down_factor)
            output_t = net(img_t)
            output_t = output_t[:,:,:H,:W]
            mask = mask[:,:,:H,:W]
            print(mask.shape)
            mask = tensor2img(mask, rgb2bgr=False, min_max=(0, 1))
            output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))

        del output_t
        torch.cuda.empty_cache()

        output = output.astype('uint8')
        # save restored img
        save_restore_path = img_path.replace(args.test_path, result_root)
        imwrite(output, save_restore_path)

    print(f'\nAll results are saved in {result_root}')
