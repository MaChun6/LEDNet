import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, win_size, dilation_rate=1):
    # x : BCHW
    B, C, H, W= x.shape
    if dilation_rate != 1:
        # x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 *
                     (dilation_rate-1), stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1,
                                                       C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
        ).view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        # B, C*Wh*Ww, H/Wh*W/Ww
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate,
                   padding=4*(dilation_rate-1), stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C //
                                 self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads,
                                         C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

class WinAttn(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., use_ca=False):
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.use_ca = use_ca
        self.scale = qk_scale or head_dim ** -0.5
                # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size - 1) * (2 * win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size)  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size)  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size - 1
        relative_coords[:, :, 0] *= 2 * self.win_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = LinearProjection(
                dim, num_heads, dim//num_heads, bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        # x: B, C, H, W
        B, C, H, W= x.shape
        x_windows = window_partition(x, self.win_size)
        x = x_windows.view(-1, self.win_size * self.win_size, C)

        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv) # B,h,N,C
        q = q * self.scale
        if not self.use_ca:
            attn = (q @ k.transpose(-2, -1)) # N * N
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size * self.win_size, self.win_size * self.win_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            ratio = attn.size(-1)//relative_position_bias.size(-1)
            relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
            attn = attn + relative_position_bias.unsqueeze(0)
        else:
            attn = (q.transpose(-2, -1) @ k)  # C * C

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        if not self.use_ca:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        else:
            x = (attn @ v.transpose(-2, -1)).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = window_reverse(
            x, self.win_size, H, W)  # B H' W' C
        x = x.permute(0, 3, 1, 2)  # B C H W
        return x

if __name__ == '__main__':
    x = torch.randn(1, 16, 256, 256)
    model = WinAttn(16, 8, 8, use_ca=True)
    y = model(x)
    print(y.shape)
