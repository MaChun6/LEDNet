
# llbnetv73
# 1. 使用Flowblork， layers=1
# 2. 最后预测mask
# 3. add csff, 将stage1的encoder和decoder

# 实验：
# 1. 用decs[i]-blocks[-i-1]，也就是1-上和2-下一起，预测失败
# 2. 用x2预测，并且调整了1-上只加到2-下,比实验1还差，完全是灰色。
# 3. 用x2和blocks[-i-1]，也就是1-上和2-下一起预测，比之前好些，感觉还是得加上blocks[-i-1]
# 4. 用x2和blocks[-i-1]，也就是1-上和2-下一起预测，使用double res。感觉没什么区别
# 5. 扩大通道数48 d到64，没啥效果
# 6. 使用单阶段的，但是并没有什么效果
# 7. 使用NAFNet block 看看，没解决
# 8. 使用多层NAFNet block，没解决
# 9. 使用NAFNet的depth4的架构，这个有点作用（4层，NAFNetblock，以及最后一层是14层）
# 10. 使用dres的depth4的架构 失败了
# 11. 使用wavelet的depth4的架构， 失败了
# 12. stage1就直接预测，不加残差  失败了
# 13. bridge 和 out concat 然后在放到里面计算mask
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from basicsr.archs.arch_util import LayerNorm2d
from basicsr.archs.module import WaveletBlock, NAFBlock, FlowBlock, FSBlock, Fusion
# MaskDeformResBlock,
def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        # img = self.conv2(x)
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

# from LEDNet
class CurveCALayer(nn.Module):
    def __init__(self, channel, n_curve):
        super(CurveCALayer, self).__init__()
        self.n_curve = n_curve
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(channel, n_curve, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # clip the input features into range of [0,1]
        a = self.predict_a(x)
        x = self.relu(x) - self.relu(x-1)
        for i in range(self.n_curve):
            x = x + a[:,i:i+1]*x*(1-x)
        return x
