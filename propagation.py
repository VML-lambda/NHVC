"""
This is the script that is used for the wave propagation using the angular spectrum method (ASM). Refer to 
Goodman, Joseph W. Introduction to Fourier optics. Roberts and Company Publishers, 2005, for principle details.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import numpy as np
import torch
import torch.fft as fft

import utils as utils


def propagation(u_in, feature_size, wavelength, z, sampling=1,
                linear_conv=False, return_H=False, precomped_H=None):
    """
        Propagates the input field using the angular spectrum method

        Inputs
        ------
        u_in:           PyTorch Complex tensor (torch.cfloat) of size (num_images, 1, height, width) -- updated with PyTorch 1.7.0 
        feature_size:   (height, width) of individual holographic features in m
        wavelength:     wavelength in m
        z:              propagation distance
        sampling:       extended band-limited ASM, default 1
        linear_conv:    if True, pad the input to obtain a linear convolution, default True
        return_H:       used for precomputing H or H_exp, ends the computation early and returns the desired variable
        precomped_H:    the precomputed value for H or H_exp

        Output
        ------
        tensor of size (num_images, 1, height, width)
    """

    # preprocess with padding for linear conv (Quadruple Extension)
    if linear_conv:
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        u_in = utils.pad_image(u_in, conv_size, padval=0, stacked_complex=False)

    if precomped_H is None:
        field_resolution = u_in.size()                              # Resolution of input field [batch, channel, height, width]
        num_y, num_x = field_resolution[-2], field_resolution[-1]   # Number of pixels
        dy, dx = feature_size                                       # Sampling interval
        y, x = (dy * float(num_y), dx * float(num_x))               # Size of the input / slm field

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)
        fxx, fyy = np.meshgrid(fx, fy)

        # transfer function (band-limited ASM)
        kernel = np.exp(1j * 2 * np.pi * z * np.sqrt(1 / wavelength**2 - (fxx**2 + fyy**2)))
        dv, du = sampling / (num_y * dy), sampling / (num_x * dx)
        bly = 1 / (wavelength * np.sqrt((2 * z * dv)**2 + 1))
        blx = 1 / (wavelength * np.sqrt((2 * z * du)**2 + 1))
        bl_filter = (np.abs(fxx) < blx) & (np.abs(fyy) < bly).astype(np.uint8)
        bl_kernel = torch.tensor(bl_filter * kernel).to(u_in.device)

    else:
        bl_kernel = precomped_H

    if return_H:
        return bl_kernel

    u_in_fft = fft.fftshift(fft.fftn(fft.fftshift(u_in), dim=(-2, -1), norm='ortho'))
    u_out_fft = bl_kernel * u_in_fft
    u_out = fft.fftshift(fft.ifftn(fft.fftshift(u_out_fft), dim=(-2, -1), norm='ortho'))
    return utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False) if linear_conv else u_out

# if __name__ == "__main__":
#     from set_dataset import get_dataloader
#     import matplotlib.pyplot as plt

#     cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
#     wavelengths = (638 * nm, 520 * nm, 450 * nm)
#     feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
#     img_res = (1072, 1920)  # resolution of SLM
#     homography_res = (880, 1600)  # for crop, see ImageLoader
#     distance = 20 * cm

#     class config:
#         def __init__(self, data_root=None):
#             self.dataset = 'DIV2K'
#             self.holo_data_root = './data'
#             self.batch_size = 2
#             self.channel = 'n'
    
#     train_config = config()
#     dataloader = get_dataloader("train", train_config)

#     for k, target in enumerate(dataloader):
#         target_amp, target_res, target_filename = target

#         if k == 0:
#             r_amp, g_amp, b_amp = torch.chunk(target_amp, 3, dim=1)
#             r_zerophase = torch.zeros_like(r_amp)
#             r_real, r_imag = utils.polar_to_rect(r_amp, r_zerophase)
#             r_target = torch.complex(r_real, r_imag)

#             slm_r = propagation(r_target, feature_size, wavelengths[2], -distance, linear_conv=True)
#             img_r = propagation(slm_r, feature_size, wavelengths[2], distance, linear_conv=True)

#             plt.subplot(1, 3, 1)
#             plt.imshow(target_amp[0][0].numpy().squeeze(), cmap='gray')
#             plt.subplot(1, 3, 2)
#             plt.imshow(np.abs(slm_r[0].numpy().squeeze()), cmap='gray')
#             plt.subplot(1, 3, 3)
#             plt.imshow(np.abs(img_r[0].numpy().squeeze()), cmap='gray')
#             plt.tight_layout()
#             plt.show()
#             exit()