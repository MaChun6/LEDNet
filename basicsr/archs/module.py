import torch
import torch.nn as nn
# from DCNv4 import DCNv4
# from mmcv.ops import ModulatedDeformConv2d, DeformConv2d
import torch.nn.functional as F
from einops import rearrange
from basicsr.archs.arch_util import LayerNorm2d
from basicsr.archs.wavelet_block import LWN
import math
# from basicsr.archs.MSCFilter import FlowModule
import numbers
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, activation=nn.LeakyReLU(0.1, inplace=True),
                 norm_layer=nn.InstanceNorm2d):
        super(ResBlock, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
                                   activation,
                                   nn.Conv2d(out_channels, in_channels, kernel_size, 1, 1))

    def forward(self, x):
        x = x + self.model(x)

        return x


class hallucination_module(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, norm_layer=nn.InstanceNorm2d):
        super(hallucination_module, self).__init__()

        self.dilation = dilation

        if self.dilation != 0:

            self.hallucination_conv = DeformConv(out_channels, out_channels, modulation=True, dilation=self.dilation)

        else:

            # self.m_conv = nn.Conv2d(in_channels, 3 * 3, kernel_size=3, stride=1, bias=True)
            # self.m_conv.weight.data.zero_()
            # self.m_conv.bias.data.zero_()
            # self.dconv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, padding=1) # DCNv2

            self.dconv = DCNv4(in_channels, kernel_size=3, padding=1, group=3, dw_kernel_size=3) # DCNv4
            self.relu = torch.nn.LeakyReLU(0.2, True)

    def forward(self, x):

        if self.dilation != 0:

            hallucination_output, hallucination_map = self.hallucination_conv(x)

        else:
            # print('before dcnv4', x.max())
            hallucination_map = 0
            if not isinstance(self.dconv, DCNv4):
                mask = torch.sigmoid(self.m_conv(x))
                offset = torch.zeros_like(mask.repeat(1, 2, 1, 1))
                hallucination_output = self.dconv(x, offset, mask)
            elif isinstance(self.dconv, DCNv4):

                b, c, h, w = x.shape
                x = rearrange(x, 'b c h w -> b (h w) c')
                hallucination_output = self.dconv(x, (h,w))
                hallucination_output = rearrange(hallucination_output, 'b (h w) c-> b c h w', h=h, w=w)
                hallucination_output = self.relu(hallucination_output)
            # print('after dcnv4', hallucination_output.max())
        return hallucination_output, hallucination_map

class MaskDeformResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm2d):
        super(MaskDeformResBlock, self).__init__()
        self.res_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
                                nn.LeakyReLU(0.1, inplace=True))
        self.hallucination_d0 = hallucination_module(out_channels, out_channels, dilation=0)

    def forward(self, x, mask=None):
        # res, map = self.res_conv(x)
        res= self.res_conv(x)
        d_out,_ = self.hallucination_d0(res)
        return d_out


class hallucination_res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=(0, 1, 2, 4), norm_layer=nn.InstanceNorm2d):
        super(hallucination_res_block, self).__init__()

        self.dilations = dilations

        self.res_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
                                      nn.LeakyReLU(0.1, inplace=True))

        self.hallucination_d0 = hallucination_module(in_channels, out_channels, dilations[0])
        self.hallucination_d1 = hallucination_module(in_channels, out_channels, dilations[1])
        self.hallucination_d2 = hallucination_module(in_channels, out_channels, dilations[2])
        self.hallucination_d3 = hallucination_module(in_channels, out_channels, dilations[3])

        self.mask_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       ResBlock(out_channels, out_channels,
                                                norm_layer=norm_layer),
                                       ResBlock(out_channels, out_channels,
                                                norm_layer=norm_layer),
                                       nn.Conv2d(out_channels, 4, 1, 1))

        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        res = self.res_conv(x)

        d0_out, _ = self.hallucination_d0(res)
        d1_out, map1 = self.hallucination_d1(res)
        d2_out, map2 = self.hallucination_d2(res)
        d3_out, map3 = self.hallucination_d3(res)

        mask = self.mask_conv(x)
        mask = torch.softmax(mask, 1)

        sum_out = d0_out * mask[:, 0:1, :, :] + d1_out * mask[:, 1:2, :, :] + \
                  d2_out * mask[:, 2:3, :, :] + d3_out * mask[:, 3:4, :, :]

        res = self.fusion_conv(sum_out) + x

        map = torch.cat([map1, map2, map3], 1)

        return res, map


class DeformConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=True, modulation=True, dilation=1):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(1)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, stride=stride, dilation=dilation,
                                padding=dilation, bias=bias)

        self.p_conv.weight.data.zero_()
        if bias:
            self.p_conv.bias.data.zero_()

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, stride=stride, dilation=dilation,
                                    padding=dilation, bias=bias)

            self.m_conv.weight.data.zero_()
            if bias:
                self.m_conv.bias.data.zero_()

            self.dconv = ModulatedDeformConv2d(inc, outc, kernel_size, padding=padding)
        else:
            self.dconv = DeformConv2d(inc, outc, kernel_size, padding=padding)

    def forward(self, x):
        offset = self.p_conv(x)

        if self.modulation:
            mask = torch.sigmoid(self.m_conv(x))
            x_offset_conv = self.dconv(x, offset, mask)
        else:
            x_offset_conv = self.dconv(x, offset)

        return x_offset_conv, offset


class Dynamic_conv(nn.Module):
    def __init__(self, kernel_size):
        super(Dynamic_conv, self).__init__()

        self.reflect_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size)

    def forward(self, x, kernel):
        b, c, h, w = x.size()
        x = self.reflect_pad(x)

        kernel = F.softmax(kernel, dim=1)

        unfolded_x = self.unfold(x)
        unfolded_x = unfolded_x.view(b, c, -1, h, w)

        out = torch.einsum('bkhw,bckhw->bchw', [kernel, unfolded_x])

        return out


class WaveletBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.wavelet_block1 = LWN(c, wavelet='haar', initialize=True)
        # self.wavelet_block1 = FFT2(c)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.wavelet_block1(x)

        x = x * self.sca(x)

        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        # gate
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma
class fft_bench_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_complex_conv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
# 部分参考 https://github.com/INVOKERer/AdaRevD/blob/master/basicsr/models/archs/AdaRevID_arch.py
class FlowBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., sin=False, window_size=8, window_size_fft=-1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft

        self.flowblock = FlowModule(c)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sg = SimpleGate()
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, z, return_mask=False):
        x= inp
        x_ = self.norm1(x)
        if return_mask:
            x, mask = self.flowblock(x_, z, return_mask=return_mask)
            # print(mask.shape)
        else:
            x = self.flowblock(x_, z)
            mask = None
        x = self.conv2(x)

        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        # fft
        x = x + self.fft_block1(x_ * mask if mask is not None else x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))
        x = self.dropout2(x)

        return y + x * self.gamma

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x, y):
        b,c,h,w = x.shape
        q = self.q(x*y) # image
        k = self.k(x) # mask
        v = self.v(x) # mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MaskCrossAttn(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False):
        super(MaskCrossAttn, self).__init__()

        self.norm1_image = LayerNorm2d(dim)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, mask):
        # image: b, c, h, w  mask: b, 1, h, w return: b, c, h, w
        mask = mask.clamp_min(0.1)
        b, c , h, w = image.shape
        fused = image + self.attn(self.norm1_image(image), mask) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused
class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
                                    act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x

class Attention(nn.Module):
    # Restormer (CVPR 2022) transposed-attnetion block
    # original source code: https://github.com/swz30/Restormer
    # 其实就是换成了dw conv
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, f):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(f))
        kv = self.kv_dwconv(self.kv_conv(x))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class DAttentionBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias):
        super(DAttentionBlock, self).__init__()

        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.da_attn = Attention(dim, num_heads, bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x = x + self.da_attn(self.norm3(x), y)
        x = x + self.ffn2(self.norm4(x))

        return x

#  https://github.com/KevinJ-Huang/FECNet.
class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nc,nc,3,1,1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return x+self.block(x)
    
class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(nc,nc,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,1,1,0))

    def forward(self,x, mode):
        mag = torch.abs(x)
        pha = torch.angle(x)
        if mode == 'amplitude':
            mag = self.process(mag)
        elif mode == 'phase':
            pha = self.process(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out

class FSBlock(nn.Module):
    def __init__(self, in_nc, dw_expand=2):
        super(FSBlock,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.norm = LayerNorm(in_nc, 'BiasFree')
        self.spatial_process1 = SpaBlock(in_nc)
        # self.frequency_process1 = FreBlock(in_nc)
        self.frequency_process1 = fft_bench_complex_conv(in_nc, dw_expand, bias=True, act_method=nn.LeakyReLU)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_process2 = SpaBlock(in_nc)
        # self.frequency_process2 = FreBlock(in_nc)
        self.frequency_process2 = fft_bench_complex_conv(in_nc, dw_expand, bias=True, act_method=nn.LeakyReLU)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.contrast = stdv_channels
        # self.process = nn.Sequential(nn.Conv2d(in_nc * 2, in_nc // 2, kernel_size=3, padding=1, bias=True),
        #                              nn.LeakyReLU(0.1),
        #                              nn.Conv2d(in_nc // 2, in_nc * 2, kernel_size=3, padding=1, bias=True),
        #                              nn.Sigmoid())

    def forward(self, x, mask=None, mask_fn=None):
        xori = x
        x = self.norm(x)
        _, _, H, W = x.shape
        if mask_fn is not None:
            mask = mask_fn(x)
        x = self.spatial_process1(x)
        x_freq = self.frequency_process1(x*mask if mask is not None else x)
        x = x+self.frequency_spatial(x_freq)
        x_freq = x_freq + self.spatial_frequency(x)
        x = self.spatial_process2(x)
        x_freq = self.frequency_process2(x_freq * mask if mask is not None else x_freq)
        xcat = torch.cat([x,x_freq],1)
        # xcat = self.process(self.contrast(xcat) + self.avgpool(xcat)) * xcat
        x_out = self.cat(xcat)
        if mask_fn  is not None:
            return x_out+xori, mask
        return x_out+xori

# https://github.com/cschenxiang/NeRD-Rain/blob/main/model.py#L202
class Fusion(nn.Module):
    def __init__(self, in_dim=32):
        super(Fusion, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x_q = self.query_conv(x)
        y_k = self.key_conv(y)
        energy = x_q * y_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_y = y * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        y_gamma = self.gamma2(torch.cat((y, attention_y), dim=1))
        y_out = y * y_gamma[:, [0], :, :] + attention_y * y_gamma[:, [1], :, :]

        x_s = x_out + y_out

        return x_s
from basicsr.archs.win_attn import WinAttn
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = WinAttn(dim, win_size, num_heads, use_ca=True)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
class FSCA(nn.Module):
    def __init__(self, in_nc, kernel_size=3, stride=1, padding=1, bias=True):
        super(FSCA, self).__init__()
        self.naf_fre = NAFBlock(in_nc)
        self.naf_pix = NAFBlock(in_nc)
        self.attn_fre = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.attn_pix = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(in_channels=in_nc * 2, out_channels=in_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        imp = x
        x_freq = self.naf_fre(x)
        x_freq = torch.fft.rfft2(x_freq, norm='backward')
        x_freq_amp = torch.abs(x_freq)
        x_freq_pha = torch.angle(x_freq)

        x_pix = self.naf_pix(x)

        x_pix = self.attn_fre(x_freq_amp) * x_pix
        x_freq_amp = self.attn_pix(x_pix) * x_freq_pha
        real = x_freq_amp * torch.cos(x_freq_pha)
        imag = x_freq_amp * torch.sin(x_freq_pha)
        x_freq = torch.complex(real, imag)
        x_freq = torch.fft.irfft2(x_freq)
        x = self.fusion(torch.cat((x_pix, x_freq), dim=1))
        return x
from mmcv.ops import ModulatedDeformConv2d, DeformConv2d
class DDDBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilations=(0, 1, 2, 4), bias=True, stride=1, use_mask=False, padding=1, use_pre_offest=False):
        super(DDDBlock, self).__init__()
        self.dilations = dilations
        self.use_mask = use_mask
        self.use_pre_offest = use_pre_offest
        self.offset_convs = nn.ModuleList([nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, 
                                                     stride=stride, dilation=dilation, padding=dilation, bias=bias) for dilation in self.dilations if dilation != 0]) 
        self.norm = LayerNorm(in_channels, 'WithBias')
        
        for module in self.offset_convs:
            module.weight.data.zero_()
            if bias:
                module.bias.data.zero_()
        if use_mask:
            self.mask_convs = nn.ModuleList([nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=3, 
                                                     stride=stride, dilation=dilation,padding=dilation, bias=bias) for dilation in self.dilations if dilation != 0]) 
            for module in self.mask_convs:
                module.weight.data.zero_()
                if bias:
                    module.bias.data.zero_()
        
        self.def_convs = nn.ModuleList([ModulatedDeformConv2d(in_channels, in_channels, kernel_size, padding=padding)])
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        if self.use_pre_offest:
            self.offset_fusions = nn.ModuleList([nn.Conv2d(2 * kernel_size * kernel_size*2, 2 * kernel_size * kernel_size, 3, 1, 1) for _ in range(len(self.dilations))])
    def forward(self, x, pre_mask=None, pre_offsets=None):
        offsets = []
        xs = []
        inp = x
        x = self.norm(x)
        for i, offset_conv in enumerate(self.offset_convs):
            offset = offset_conv(x)
            offsets.append(offset)
        if self.use_mask:
            masks = []
            for i, mask_conv in enumerate(self.mask_convs):
                mask = torch.sigmoid(mask_conv(x))
                masks.append(mask)
        for i, def_conv in enumerate(self.def_convs):
            x_def = def_conv(x, offsets[i] if not self.use_pre_offest else self.offset_fusions[i](torch.cat([offsets[i], F.interpolate(pre_offsets[i], scale_factor=2)], dim=1)),
                              pre_mask if not self.use_mask else masks[i])
            xs.append(x_def)
        xs = torch.stack(xs)
        xs = torch.sum(xs, dim=0)
        x = self.fusion_conv(xs)*pre_mask + inp * (1-pre_mask)
        return x
if __name__ == '__main__':
    model = TransformerBlock(64, 4).cuda()
    x = torch.randn(1, 64, 256, 256).cuda()
    out = model(x)
    print(out.shape)
