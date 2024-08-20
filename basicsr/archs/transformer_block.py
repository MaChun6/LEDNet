from basicsr.archs.win_attn import WinAttn
from basicsr.archs.MSCFilter_sub import BasicConv_do, ResBlock_do_fft_bench, ResBlock_do_fft_bench_eval, BasicConv_do_eval
import torch_dct as dct

class CAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SFTFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(SFTFFN, self).__init__()
        hidden_features = int(dim)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.se = SELayer(dim)
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x_freq = torch.fft.rfft2(x1, norm='backward')
        x_mag = torch.abs(x_freq)
        x_pha = torch.angle(x_freq)
        x_mag = self.se(x_mag)
        real = x_mag * torch.cos(x_pha)
        imag = x_mag * torch.sin(x_pha)
        x_fft = torch.complex(real, imag)
        x2 = torch.fft.irfft2(x_fft, s=(H, W), norm='backward')
        # x1 和x2该怎么选
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class CTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(CTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = SFTFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        B,C,H,W = x.shape
        x_freq = torch.fft.rfft2(x, dim=[-2, -1], norm='backward')
        x_mag = torch.abs(x_freq)
        x_pha = torch.angle(x_freq)
        x_mag = self.attn(self.norm1(x_mag))
        real = x_mag * torch.cos(x_pha)
        imag = x_mag * torch.sin(x_pha)
        x_freq = torch.complex(real, imag)
        x = torch.fft.irfft2(x_freq, s=(H, W), norm='backward') + x
        x = x + self.ffn(self.norm2(x))

        return x
