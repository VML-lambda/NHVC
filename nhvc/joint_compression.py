import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.models.utils import update_registered_buffers
from compressai.models import (
    get_scale_table,
    MeanScaleHyperprior, 
    CompressionModel
)

from nhvc.compression.layers import conv3x3, upsample_scale_2, DeformableConv2d
from nhvc.compression.module import (
    ResidualBlockDeformable, 
    ResidualBlockWithStride,
    ResidualBlockUpsample
)


class HoloCompression(MeanScaleHyperprior):
    """
        Scale Hyperprior with non zero-mean Gaussian conditionals from D.
        Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
        Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
        Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

        HoloCompression initialization parameters
        -------------------------------
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
                 encoder and last layer of the hyperprior decoder)
    """
    def __init__(self, N=128, gain=True, **kwargs):
        super(HoloCompression, self).__init__(N=N, M=N, **kwargs)
        num_in = 2
        num_out = 1

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(num_in, N, stride=2),
            ResidualBlockDeformable(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlockDeformable(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlockDeformable(N, N),
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=1)
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            upsample_scale_2(N, N, upsample_mode="transposed"),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N*3//2),
            nn.LeakyReLU(inplace=True),
            upsample_scale_2(N*3//2, N*3//2, upsample_mode="transposed"),
            nn.LeakyReLU(inplace=True),
            conv3x3(N*3//2, N*2)
        )

        self.g_s = nn.Sequential(
            ResidualBlockDeformable(N, N),
            ResidualBlockUpsample(N, N, upsample_mode="transposed"),
            ResidualBlockDeformable(N, N),
            ResidualBlockUpsample(N, N, upsample_mode="transposed"),
            ResidualBlockDeformable(N, N),
            ResidualBlockUpsample(N, N, upsample_mode="transposed"),
            ResidualBlockDeformable(N, N),
            upsample_scale_2(N, num_out, upsample_mode="transposed"),
            nn.Hardtanh(-torch.pi, torch.pi)
        )

        self.gain = gain
        # self.lmbda = [0.05, 0.025, 0.015, 0.008, 0.004, 0.002]  # low gain
        # self.lmbda = [0.3, 0.1, 0.05, 0.025, 0.015, 0.008] # mid gain
        self.lmbda = [4, 1, 0.3, 0.1, 0.05, 0.03]  # high gain
        
        # self.lmbda = [0.15, 0.08, 0.04, 0.015, 0.006, 0.002]

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        self.levels = len(self.lmbda)
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)


    def generate_forward(self, complex):
        latent_map = self.g_a(complex)
        # latent_map_hat, _ = self.entropy_bottleneck(latent_map)
        output_poh = self.g_s(latent_map)
        return {
            "latent_map": latent_map, 
            "output_poh": output_poh
        }


    def compress_forward(self, complex, s):
        y = self.g_a(complex)
        if self.gain:
            y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z = self.h_a(y)
        if self.gain:
            z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.gain:
            z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        if self.gain:
            y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x_hat = self.g_s(y_hat)
        return {
            "latent_map": y, 
            "output_poh": x_hat,
            "likelihoods": {
                "y": y_likelihoods, 
                "z": z_likelihoods
            }
        }


    def compress(self, complex, s):
        if self.gain:
            assert s in range(0,self.levels), f"s should in range(0, {self.levels}), but get s:{s}"

        y = self.g_a(complex)   # Latent feature map
        ungained_y = y
        if self.gain:
            y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z = self.h_a(y)         # Hyperprior feature map
        if self.gain:
            z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z_strings = self.entropy_bottleneck.compress(z)                         # Hyperprior "compress"
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])    # Hyperprior decompress
        if self.gain:
            z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        gaussian_params = self.h_s(z_hat)                   
        scales_hat, means_hat = gaussian_params.chunk(2, 1) 
        indexes = self.gaussian_conditional.build_indexes(scales_hat)               # Set Gaussian Scale
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat) # Latent "compress"
        # gained_y_hat = self.gaussian_conditional.quantize(y, "symbols", means_hat)
        return {
            "latent_map": y,
            "strings": [y_strings, z_strings], 
            "shape": z.size()[-2:],
            "ungained_y_hat": ungained_y,
        }
    

    def decompress(self, compressed, s):
        strings = compressed["strings"]
        shape = compressed["shape"]
        assert isinstance(strings, list) and len(strings) == 2
        
        if self.gain:
            assert s in range(0, self.levels), f"s should in range(0,{self.levels})"

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)     # Hyperprior decompress
        if self.gain:
            z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)    # Latent feature map decompress
        gained_y_hat = y_hat
        if self.gain:
            y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x_hat = self.g_s(y_hat) # Phase-only Hologram
        return {
            "output_poh": x_hat,
            "gained_y_hat": gained_y_hat,
            "y_hat": y_hat
        }


    def getY(self, x, s):
        assert s in range(0, self.levels), f"s should in range(0, {self.levels - 1}), but got s:{s}"
        y = self.g_a(x)
        ungained_y = y
        if self.gain:
            y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_quantized = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")

        return {"ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": y_quantized}
    
    def getScale(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        gaussian_params = self.h_s(z)
        scales, means = gaussian_params.chunk(2, 1)
        return scales

    def getX(self, y_hat, s):
        assert s in range(0, self.levels), f"s should in range(0,{self.levels - 1})"

        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat)
        return {"output_poh": x_hat}
    


class SpatioTemporalHyperpriorCoder(CompressionModel):
    """
    One of Ablation Experiments for SpatioTemporalPriorModel.
    Temporal Prior Model contains hyperprior and temporal prior but not spatial prior.
    All prior information is used to estimate gaussian probability model parameters, thus has a GaussianConditional model.
    Model inherits from CompressionModel which contains an entropy_bottleneck used for compress hyperprior latent z.
    """

    def __init__(self, entropy_bottleneck_channels=128, N=128, gain=True):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        self.gaussian_conditional = GaussianConditional(None)
        self.TPM = nn.Sequential(
            DeformableConv2d(N, N*3//2),
            nn.LeakyReLU(),
            DeformableConv2d(N*3//2, N*2),
            nn.LeakyReLU(),
            DeformableConv2d(N*2, N*3),         
        )
        self.PEh = nn.Sequential(
            conv3x3(N * 2, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.PDh = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            upsample_scale_2(N, N, upsample_mode="transposed"),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N*3//2),
            nn.LeakyReLU(inplace=True),
            upsample_scale_2(N*3//2, N*3//2, upsample_mode="transposed"),
            nn.LeakyReLU(inplace=True),
            conv3x3(N*3//2, N*2)
        )
        self.EPM = nn.Sequential(
            nn.Conv2d(in_channels=N*5, out_channels=N*4, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=N*4, out_channels=N*3, kernel_size=1, padding=1 // 2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=N*3, out_channels=N*2, kernel_size=1, padding=1 // 2, stride=1),
        )
        self.levels = 6
        self.gain = gain
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

    # def forward(self, y_cur, y_conditioned, s):
    def forward(self, y_cur, y_conditioned, s=0):
        z = self.PEh(torch.cat([y_cur, y_conditioned], 1))
        if self.gain:
            z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.gain:
            z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        hyperprior_params = self.PDh(z_hat)
        temporalprior_params = self.TPM(y_conditioned)
        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y_cur, scales_hat, means=means_hat) # used res_val
        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, y_cur, y_conditioned, s):
        z = self.PEh(torch.cat([y_cur, y_conditioned], 1))
        if self.gain:
            z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        if self.gain:
            z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        hyperprior_params = self.PDh(z_hat)

        temporalprior_params = self.TPM(y_conditioned)

        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y_cur, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, y_conditioned, s):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        if self.gain:
            z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        hyperprior_params = self.PDh(z_hat)

        temporalprior_params = self.TPM(y_conditioned)
        gaussian_params = self.EPM(torch.cat([temporalprior_params, hyperprior_params], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return {"y_hat": y_hat}


    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


if __name__ == "__main__":
    
    model = SpatioTemporalHyperpriorCoder()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of parameters: {total_params}")
    #from torchsummary import summary
    #model = HoloCompression(128, True)
    #summary(model.cuda(), (3, 1024, 1024))
