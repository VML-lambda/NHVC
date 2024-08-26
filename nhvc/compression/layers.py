import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from compressai.ops.parametrizers import NonNegativeParametrizer


def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def upsample_scale_2(in_ch, out_ch, upsample_mode="transposed"):
    if upsample_mode == "transposed":
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, output_padding=0, padding=1)
    
    elif upsample_mode == "pixel_shuffle":
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 2**2, kernel_size=3, padding=1), 
            nn.PixelShuffle(2)
        )
    
    else:   # mode == "nearest", "linear", "bilinear", "bicubic", "trilinear"
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), 
            nn.Upsample(scale_factor=2, mode=upsample_mode),
        )


class DeformableConv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2*kernel_size*kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, 1*kernel_size*kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator
        )


class GDN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inverse: bool = False,
                 beta_min: float = 1e-6,
                 gamma_init: float = 0.1):
        super(GDN, self).__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)
        norm = torch.sqrt(norm) if self.inverse else torch.rsqrt(norm)

        out = x * norm
        return out
     

if __name__ == "__main__":
    from torchsummary import summary
    model = DeformableConv2d(3, 16)
    summary(model.cuda(), (3, 16, 16))
