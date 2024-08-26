import torch
import torch.nn as nn
from torchsummary import summary


class InitialPhaseUnet(nn.Module):
    """
        Computes the initial input phase given a target amplitude
    
        InitialPhaseUnet initialization parameters
        -------------------------------
        num_down:           Number of downsampling stages.
        num_features_init:  Number of features at highest level of U-Net
        max_features:       Maximum number of channels (channels multiply by 2 with every downsampling stage)
        norm:               Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        fullcolor:          If True, generate initial phase of blue, green, red channel

        Returns
        -------------------------------
        out_phase:          Initial Phase (~θ, θ)
    """
    def __init__(self, 
                 num_down=4, 
                 num_features_init=16, 
                 max_features=256, 
                 norm=nn.BatchNorm2d):
        super(InitialPhaseUnet, self).__init__()
        channel = 1

        net = nn.Sequential(Unet(channel, channel, 
                                 num_features_init, 
                                 max_features,
                                 num_down, 
                                 upsampling_mode='transpose',
                                 use_dropout=False,
                                 norm=norm,
                                 outermost_linear=True),
                            nn.Hardtanh(-torch.pi, torch.pi))
        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase
    

class DownBlock(nn.Module):
    """
        A 2D-conv downsampling block following best practices / with reasonable defaults 
        (LeakyReLU, kernel size multiple of stride)

        DownBlock initialization parameters
        -------------------------------
        in_channels:        Number of input channels
        out_channels:       Number of output channels
        prep_conv:          Whether to have another convolutional layer before the downsampling layer.
        middle_channels:    If prep_conv is true, this sets the number of channels between the prep and downsampling convs.
        use_dropout:        bool. Whether to use dropout or not.
        dropout_prob:       Float. The dropout probability (if use_dropout is True)
        norm:               Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 prep_conv=True,
                 middle_channels=None,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d):
        super(DownBlock, self).__init__()
        middle_channels = in_channels if middle_channels is None else middle_channels
        bias=True if norm is None else False
        net = list()

        # Convolution Layer
        if prep_conv:
            net += [nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
            if norm is not None:
                net += [norm(middle_channels, affine=True)]
            net += [nn.LeakyReLU(0.2, True)]
            if use_dropout:
                net += [nn.Dropout2d(dropout_prob, False)]

        # Down-sampling
        net += [nn.Conv2d(middle_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=bias)]
        if norm is not None:
            net += [norm(out_channels, affine=True)]
        net += [nn.LeakyReLU(0.2, True)]
        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        # Final Down-block network
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
    

class UpBlock(nn.Module):
    """
        A 2d-conv upsampling block with a variety of options for upsampling, and following best practices / with reasonable defaults. 
        (LeakyReLU, kernel size multiple of stride)

        UpBlock initialization parameters
        -------------------------------
        in_channels:        Number of input channels
        out_channels:       Number of output channels
        post_conv:          Whether to have another convolutional layer after the upsampling layer.
        use_dropout:        bool. Whether to use dropout or not.
        dropout_prob:       Float. The dropout probability (if use_dropout is True)
        norm:               Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        upsampling_mode:    Which upsampling mode:
                            - transpose(default): Upsampling with stride-2, kernel size 4 transpose convolutions.
                            - bilinear: Feature map is upsampled with bilinear upsampling, then a conv layer.
                            - nearest: Feature map is upsampled with nearest neighbor upsampling, then a conv layer.
                            - shuffle: Feature map is upsampled with pixel shuffling, then a conv layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 post_conv=True,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 upsampling_mode='transpose'):
        super(UpBlock, self).__init__()
        bias=True if norm is None else False
        net = list()

        # Up-sampling
        if upsampling_mode == 'transpose':
            net += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias)]
        elif upsampling_mode == 'bilinear':
            net += [nn.UpsamplingBilinear2d(scale_factor=2)]
            net += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
        elif upsampling_mode == 'nearest':
            net += [nn.UpsamplingNearest2d(scale_factor=2)]
            net += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
        elif upsampling_mode == 'shuffle':
            net += [nn.PixelShuffle(upscale_factor=2)]
            net += [nn.Conv2d(in_channels // 4, out_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
        else:
            raise ValueError("Unknown upsampling mode!")
        if norm is not None:
            net += [norm(out_channels, affine=True)]
        net += [nn.ReLU(True)]
        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        # Convolution Layer
        if post_conv:
            net += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
            if norm is not None:
                net += [norm(out_channels, affine=True)]
            net += [nn.ReLU(True)]
            if use_dropout:
                net += [nn.Dropout2d(0.1, False)]

        # Final Up-block network
        self.net = nn.Sequential(*net)

    def forward(self, x, skipped=None):
        # if skipped is not None:
        #     input = torch.cat([skipped, x], dim=1)
        # else:
        #     input = x
        input = torch.cat([skipped, x], dim=1) if skipped is not None else x
        return self.net(input)
    

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 outer_nc,
                 inner_nc,
                 upsampling_mode,
                 norm=nn.BatchNorm2d,
                 submodule=None,
                 use_dropout=False,
                 dropout_prob=0.1):
        super(UnetSkipConnectionBlock, self).__init__()

        if submodule is None:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     UpBlock(inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm, upsampling_mode=upsampling_mode)]
        else:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     submodule,
                     UpBlock(2 * inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm, upsampling_mode=upsampling_mode)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return torch.cat([x, forward_passed], 1)


class Unet(nn.Module):
    """
        A 2d-Unet implementation with sane defaults

        U-net initialization parameters
        -------------------------------
        in_channels:        Number of input channels
        out_channels:       Number of output channels
        nf0:                Number of features at highest level of U-Net
        max_channels:       Maximum number of channels (channels multiply by 2 with every downsampling stage)
        num_down:           Number of downsampling stages.
        use_dropout:        Whether to use dropout or no.
        dropout_prob:       Dropout probability if use_dropout=False.
        upsampling_mode:    Which type of upsampling should be used. See "UpBlock" for documentation.
        norm:               Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        outermost_linear:   Whether the output layer should be a linear layer or a nonlinear one.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 max_channels,
                 num_down,                 
                 upsampling_mode='transpose',
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 outermost_linear=True):
        super(Unet, self).__init__()
        assert (num_down > 0), "Need at least one downsampling layer in UNet."

        # Define the in-block
        self.in_layer = [nn.Conv2d(in_channels, nf0, kernel_size=3, padding=1, stride=1, bias=True if norm is None else False)]
        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]
        self.in_layer += [nn.LeakyReLU(0.2, True)]
        if use_dropout:
            self.in_layer += [nn.Dropout2d(dropout_prob)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center U-net block
        self.unet_block = UnetSkipConnectionBlock(min(2 ** (num_down-1) * nf0, max_channels), 
                                                  min(2 ** (num_down-1) * nf0, max_channels),
                                                  use_dropout=use_dropout, dropout_prob=dropout_prob,
                                                  norm=None, # Innermost has no norm (spatial dimension 1)
                                                  upsampling_mode=upsampling_mode)
        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),  
                                                      min(2 ** (i + 1) * nf0, max_channels),
                                                      use_dropout=use_dropout, dropout_prob=dropout_prob,
                                                      submodule=self.unet_block,
                                                      norm=norm,
                                                      upsampling_mode=upsampling_mode)

        # Define the out-layer
        self.out_layer = [nn.Conv2d(2 * nf0, out_channels, kernel_size=3, padding=1, stride=1, bias=outermost_linear or (norm is None))]
        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.ReLU(True)]
            if use_dropout:
                self.out_layer += [nn.Dropout2d(dropout_prob)]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer
    

if __name__ == "__main__":
    net = InitialPhaseUnet(4, 16).to("cuda")
    summary(net, (3, 512, 512))
