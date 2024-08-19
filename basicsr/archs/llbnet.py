
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



class FreMaskBlock(nn.Module):
    def __init__(self, nc):
        super(FreMaskBlock, self).__init__()
        self.nc = nc
        self.in_conv = nn.Sequential(
            LayerNorm2d(nc),
            nn.Conv2d(nc,nc,1,1,0),
            nn.GELU()
        )
        self.mask_pre = nn.Sequential(
            nn.Conv2d(nc,nc//2,1,1,0),
            nn.GELU(),
            nn.Conv2d(nc//2,nc//4,1,1,0),
            nn.GELU(),
            nn.Conv2d(nc//4,1,1,1,0)
        )
        # self.log = nn.LogSoftmax(dim=-1)
        self.log = nn.Sigmoid()
    def forward(self, dec, enc=None):
        b,c,h,w = dec.shape
        if enc is not None:
            x = self.in_conv(dec+enc)
        else:
            x = self.in_conv(dec)
        local_x = x[:, :c//2, :, :]
        global_x = torch.mean(x[:, c//2:, :, :], keepdim=True, dim=(2,3))
        x = torch.cat([local_x, global_x.expand(b, c//2, h, w )], dim=1)
        x = self.mask_pre(x)
        mask = self.log(x)
        # x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w)

        # pred_score = self.log(x)
        # mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]
        # mask = rearrange(mask, 'b (h1 w1) c -> b c h1 w1', h1=h, w1=w)
        return mask


# @ARCH_REGISTRY.register()
class LLBNetv74(nn.Module):
    def __init__(self, in_chn=3, wf=48, depth=4, fuse_before_downsample=True, relu_slope=0.2, num_heads=[4,8,0], layers=[1,1,1,1]):
        super(LLBNetv74, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.outconvs = nn.ModuleList([conv3x3((2 ** i)* wf, in_chn, bias=True) for i in reversed(range(depth-1)) if i !=0 ])
        self.layers = layers
        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, block_type='res'))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, block_type='res', layers=self.layers[i], use_csff=True))
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        self.mask_pred = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope, use_curve=True, block_type='res'))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope, block_type='fs', layers=self.layers[i]))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.mask_pred.append(FreMaskBlock((2**(i))*wf))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x):
        image = x
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        xs = [x_4, x_2, x]
        #stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        outs = []
        # 1 - 下
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1, merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)
            else:
                x1 = down(x1, merge_before_downsample=self.fuse_before_downsample)

        # 1 -上
        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)

        x2_2 = F.interpolate(out_1, scale_factor=0.5)
        x2_4 = F.interpolate(x2_2, scale_factor=0.5)
        x2s = [x2_4, x2_2, out_1]

        #stage 2
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        # 2 - 下
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i-1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        # 2 - 上
        for i, (up, mask_pre) in enumerate(zip(self.up_path_2, self.mask_pred)):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]), xs=xs[i])
            mask = mask_pre(x2).clamp_min(0.1)
            if i != self.depth-2:
                _img = self.outconvs[i](x2) * mask + xs[i]
                outs.append(_img)
            masks.append(mask)

        out_2 = self.last(x2)
        mask = mask_pre(x2).clamp_min(0.1)
        out_2 = out_2 * mask + xs[-1]
        outs.append(out_2)
        masks = masks[::-1]
        if self.training:
            return {'stage1' : [out_1], 'stage2':outs[::-1], 'masks': masks}
        else:
            return out_2, masks[0]
        # return [out_1, out_2]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None, use_curve=False, block_type='res', layers=1, use_csff=False): # cat
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.use_emgc = use_emgc
        self.num_heads = num_heads
        self.use_curve = use_curve
        self.block_type = block_type
        self.use_csff = use_csff
        # Res 和 Dres 缺少那种膨胀操作，类似于FFN里面那种C->2C->C的操作
        if self.block_type=='res':
            self.upchannel = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
            self.res = nn.ModuleList([nn.Sequential(LayerNorm2d(out_size),
                                    nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(relu_slope, inplace=False),
                                    nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
                                   )])

        elif block_type=='wavelet':
            # self.res = MaskDeformResBlock(in_size, out_size)
            self.upchannel = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
            self.res = nn.ModuleList([nn.Sequential(*[WaveletBlock(out_size) for _ in range(layers)])])

        elif block_type=='dres':
            self.upchannel = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
            self.res = nn.ModuleList([nn.Sequential(LayerNorm2d(out_size),
                                    nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(relu_slope, inplace=False),
                                    nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
                                   ),
                                   nn.Sequential(LayerNorm2d(out_size),
                                    nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(relu_slope, inplace=False),
                                    nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
                                   ),])
        elif block_type=='nafnet':
            self.upchannel = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
            self.res = nn.ModuleList([nn.Sequential(*[NAFBlock(out_size) for _ in range(layers)])])

        elif block_type=='flow':
            self.upchannel = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
            self.res = nn.ModuleList([FlowBlock(in_size) for _ in range(layers)])

        elif block_type=='fs':
            self.upchannel = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
            self.res = nn.ModuleList([FSBlock(in_size) for _ in range(layers)])

        if self.use_curve:
            self.curve = CurveCALayer(out_size, 3)

        if downsample and use_csff:
            # print('use_csff')
            # self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            # self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff = Fusion(in_dim = out_size)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        # if self.num_heads is not None and self.num_heads!=0:
        #     self.mask_crossattn_enc = MaskCrossAttn(out_size, num_heads=self.num_heads, ffn_expansion_factor=2, bias=False)
        #     self.mask_crossattn_dec = MaskCrossAttn(out_size, num_heads=self.num_heads, ffn_expansion_factor=2, bias=False)


    def forward(self, x, enc=None, dec=None, mask=None, merge_before_downsample=True, xs=None):
        if 'res' in self.block_type:
            x= self.upchannel(x)
            for i, blk in enumerate(self.res):
                x = blk(x) + x
        elif 'flow' in self.block_type:
            for i, blk in enumerate(self.res):
                x = blk(x, xs, return_mask=True)
            x= self.upchannel(x)
        else:
            for i, blk in enumerate(self.res):
                x = blk(x)
            x= self.upchannel(x)

        out = x

        if self.use_curve:
            out = self.curve(out)

        if enc is not None and dec is not None:
            # assert self.use_csff
            # out = out + self.csff_enc(enc) + self.csff_dec(dec)
            assert self.use_csff
            out = out + self.csff(enc, dec)

        if self.num_heads is not None and self.num_heads!=0:
            # out_enc = self.mask_crossattn_enc(out, mask)
            # out_dec = self.mask_crossattn_dec(dec, 1-mask)
            out = out * mask + dec * (1-mask)

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out

        else:
            if merge_before_downsample:
                return out

class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope, use_curve=False, block_type=False, num_heads=None, layers=1):
        super(UNetUpBlock, self).__init__()
        self.num_heads = num_heads
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope, use_curve=use_curve, block_type=block_type, num_heads=num_heads, layers=layers)
        # if self.num_heads is not None and self.num_heads!=0:
        #     self.mask_crossattn_enc = MaskCrossAttn(out_size, num_heads=self.num_heads, ffn_expansion_factor=2, bias=False)
        #     self.mask_crossattn_dec = MaskCrossAttn(out_size, num_heads=self.num_heads, ffn_expansion_factor=2, bias=False)

    def forward(self, x, bridge, dec=None, mask=None, mask_fn=None, xs=None):
        up = self.up(x)
        # if dec is not None and mask is not None:
        #     up = self.mask_crossattn_enc(up, mask)
        #     dec = self.mask_crossattn_dec(dec, 1-mask)
        #     up = up + dec
        if mask is not None:
            out = torch.cat([up, bridge], 1)
        else:
            out = torch.cat([up, bridge], 1)

        if mask_fn is not None:
            mask = mask_fn(out)
            out = out * mask
        out = self.conv_block(out, xs=xs)
        if mask_fn is None:
            return out
        else:
            return out, mask


if __name__ == "__main__":
    model = LLBNetv74().cuda()
    model.train()
    x = torch.randn(1, 3, 256, 256).cuda()
    mask = torch.randn(1, 1, 256, 256).cuda()
    if model.training:
        out = model(x)
        for stage1 in out['stage1']:
            print(stage1.shape)
        for stage2 in out['stage2']:
            print(stage2.shape)
        for mask in out['masks']:
            print(mask.shape)
    else:
        out, mask = model(x)
        print('Output shape:', mask.shape)

    print('# model_restoration parameters: %.2f M' % (sum(param.numel()
          for param in model.parameters()) / 1e6))
    for i, name in enumerate(model.state_dict()):
        print(f'[{i}] {name}')

    # i = 0
    # for k, v in model.state_dict().items():
    #     print(f'[{i}]: {k}')
    #     i+=1
    # model = CurveChebyLayer(3,4).cuda()
    # x = torch.randn(1, 3, 256, 256).cuda()
    # out = model(x)
    # print(out.shape)
