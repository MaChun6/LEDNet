
def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device

    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
            indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output

class FlowModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=7):
        super(FlowModule, self).__init__()
        if self.training:
            BasicConv = BasicConv_do
            ResBlock = ResBlock_do_fft_bench
        else:
            BasicConv = BasicConv_do_eval
            ResBlock = ResBlock_do_fft_bench_eval

        self.kernel_size = kernel_size
        self.kernel_pad = int((self.kernel_size - 1) / 2.0)
        self.dim = in_channels

        self.KernelPredictFlow=BasicConv(in_channels, 2, kernel_size=3, relu=False, stride=1)
        self.KernelPredictFlowMask = BasicConv(in_channels, 1, kernel_size=3, relu=False, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv = BasicConv(in_channels *2 , in_channels, kernel_size=3, relu=False, stride=1)

    def forward(self, z, x, return_mask=False):
        kernal_flow = self.KernelPredictFlow(z)
        kernal_flowmask = self.KernelPredictFlowMask(z)
        kernal_flowmask = self.sigmoid(kernal_flowmask)

        zx = torch.cat([z,x],1)
        kernal_flowfeat0, x_0 = torch.split(flow_warp(zx, kernal_flow.permute(0, 2, 3, 1)), self.dim , dim=1)
        kernal_flowfeat1, x_1 = torch.split(flow_warp(zx, -kernal_flow.permute(0, 2, 3, 1)), self.dim , dim=1)
        # x_4 = x_0 * kernal_flowmask + x_1 * (1-kernal_flowmask)
        # 这里可以交叉注意力
        z = torch.cat([z, kernal_flowfeat0 * kernal_flowmask + kernal_flowfeat1 * (1-kernal_flowmask)],1)
        z = self.conv(z)
        if return_mask:
            return z, kernal_flowmask
        else:
            return z

class FGDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FGDFN, self).__init__()

        # hidden_features = int(dim*ffn_expansion_factor)
        hidden_features = int(dim)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.se = SELayer(dim)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x_dct = dct.dct_2d(x1)
        x_dct = self.se(x_dct)
        x1 = dct.idct_2d(x_dct)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FLowTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',use_flow=False):
        super(FLowTransformerBlock, self).__init__()
        self.use_flow=use_flow
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = WinAttn(dim, win_size, num_heads, use_ca=True)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FGDFN(dim, ffn_expansion_factor, bias)
        if use_flow:
            self.flow_module = FlowModule(dim)

    def forward(self, x, z=None):
        if self.use_flow:
            x = self.flow_module(x, z, return_mask=False)
        x_dct= dct.dct_2d(x).contiguous()
        x_dct = self.attn(self.norm1(x_dct))
        x =  x + dct.idct_2d(x_dct)
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == '__main__':
    model = FLowTransformerBlock(64, 4).cuda()
    # import torch_dct as dct
    x = torch.randn(1, 64, 256, 256).cuda()
    z = torch.randn(1, 3, 256, 256).cuda()
    out = model(x, z)
    print(out.shape)
