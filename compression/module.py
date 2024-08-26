from torch import Tensor
import torch.nn as nn

from nhvc.compression.layers import conv3x3, conv1x1, upsample_scale_2
from nhvc.compression.layers import DeformableConv2d, GDN


class ResidualBlockDeformable(nn.Module):
    """
        Residual deformable block with a stride on the first convolution.

        ResidualBlockDeformable initialization parameters
        -------------------------------
        in_ch (int):    number of input channels
        out_ch (int):   number of output channels
        stride (int):   stride value (default: 2)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super(ResidualBlockDeformable, self).__init__()
        self.conv1 = DeformableConv2d(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.conv2 = conv3x3(out_ch, out_ch)
        # self.gdn = GDN(out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else None
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        # out = self.gdn(out)

        identity = x if self.skip is None else self.skip(x)
        out += identity
        return out


class ResidualBlockWithStride(nn.Module):
    """
        Residual block with a stride on the first convolution.

        ResidualBlockWithStride initialization parameters
        -------------------------------
        in_ch (int):    number of input channels
        out_ch (int):   number of output channels
        stride (int):   stride value (default: 2)
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super(ResidualBlockWithStride, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        self.skip = conv1x1(in_ch, out_ch, stride) if stride != 1 or in_ch != out_ch else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        identity = x if self.skip is None else self.skip(x)
        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """
        Residual block with upsampling on the last convolution.

        ResidualBlockUpsample initialization parameters
        -------------------------------
        in_ch (int):    number of input channels
        out_ch (int):   number of output channels
        upsample (str): upsampling mode (transposed, pixel_shuffle, nearest, bilinear, bicubic)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample_mode: str="transposed"):
        super(ResidualBlockUpsample, self).__init__()
        self.subpel_conv1 = upsample_scale_2(in_ch, out_ch, upsample_mode)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.skip = upsample_scale_2(in_ch, out_ch, upsample_mode)

    def forward(self, x: Tensor) -> Tensor:
        out = self.subpel_conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.igdn(out)
        identity = self.skip(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
