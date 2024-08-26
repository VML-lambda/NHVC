import os
import math
import random
import cv2

import torch
from torch import nn, optim
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar as RICH
from pytorch_lightning.loggers import WandbLogger
# from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.strategies import DDPStrategy

from config import Config
from nhvc.init_phase_generation import InitialPhaseUnet
from nhvc.joint_compression import HoloCompression
from set_lightning_datamodule import LightningDataModule
from utils import crop_image, phasemap_8bit, pad_image
from propagation import propagation


# Setting up the environment variable for Weights & Biases API Key
os.environ["WANDB_API_KEY"] = ""
torch.set_float32_matmul_precision('high')

class ImageCompressionModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False

        self.image_compression = config.image_compression
        # Train generation / compression
        self.is_train = config.train
        self.gain = config.gain

        # SLM propagation configuration
        cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
        self.channel = config.channel
        self.fullcolor = True if config.channel == 'n' else False
        self.wavelength = {'b': 450 * nm, 'g': 520 * nm, 'r': 638 * nm}
        self.featuresize = (config.featuresize * um, config.featuresize * um)
        self.propdist = config.propdist * cm
        self.image_res = config.image_res
        self.homography_res = config.homography_res

        # Initial Phase Generation Network
        self.initial_phase_generation = InitialPhaseUnet(
            num_down=4,
            num_features_init=16,
            max_features=256,
            norm=nn.BatchNorm2d
        )

        # Compression Module
        self.holo_compression = HoloCompression(N=128, gain=self.gain)

        # Training configuration
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.aux_lr = config.aux_lr
        self.lr_step_size = config.lr_step_size
        self.criterionMSE = nn.MSELoss()

        # Metrics configuration
        self.psnr = PSNR(data_range=1.0, reduction='elementwise_mean')
        self.ssim = SSIM(gaussian_kernel=True, sigma=1.5, kernel_size=11, reduction='elementwise_mean', data_range=1.0, k1=0.01, k2=0.03)
        self.save_hyperparameters()



    # Set optimizer
    def configure_optimizers(self):
        models = [self.initial_phase_generation, self.holo_compression]

        parameters = {n for model in models for n, p in model.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
        aux_parameters = {n for model in models for n, p in model.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
        params_dict = {n: p for model in models for n, p in model.named_parameters()}

        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters
        assert len(inter_params) == 0
        # assert len(union_params) - len(params_dict.keys()) == 0
        
        optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=self.lr)
        aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=self.aux_lr)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
        return [optimizer, aux_optimizer], [lr_scheduler]
    

    # Generate Initial Hologram (SLM plane)
    def initial_hologram_generation(self, x):
        init_phase = self.initial_phase_generation(x)
        init_complex = x * torch.exp(1j * init_phase)

        # Propagation Target to SLM
        slm = propagation(init_complex, self.featuresize, self.wavelength[self.channel], -self.propdist)
        slm_amp_phase = torch.cat((torch.abs(slm), torch.angle(slm)), dim=1).float()

        return slm_amp_phase
    

    # Stage 1
    def stage_1(self, slm_amp_phase):
        output = self.holo_compression.generate_forward(slm_amp_phase)
        return output
    

    # Stage 2 
    def stage_2(self, slm_amp_phase, s):
        if self.is_train:
            output = self.holo_compression.compress_forward(slm_amp_phase, s)
            return output

        else: # For Testing
            compressed = self.holo_compression.compress(slm_amp_phase, s)
            decompressed = self.holo_compression.decompress(compressed, s)
            output = compressed | decompressed
            return output


    def run_step(self, target, mask, s=0):
        # Generate Initial complex hologram
        slm_amp_phase = self.initial_hologram_generation(target)

        # If stage 1: Generate Final phase-only hologram straightforward.
        if not self.image_compression:
            output = self.stage_1(slm_amp_phase)

        # If stage 2: Generate Final phase-only hologram with latent compression.
        elif self.image_compression:
            output = self.stage_2(slm_amp_phase, s)

        # Reconstruct Final phase-only hologram
        output_amp = self.reconstruction(output["output_poh"]) * 0.95
        output["output_amp"] = output_amp * mask
        output["target_amp"] = target * mask

        return output


    #############################
    #       Training Step       #
    #############################
    def training_step(self, data, data_idx):
        optimizer, aux_optimizer = self.optimizers()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        s = random.randint(0, self.holo_compression.levels - 1)
        output = self.run_step(data['target'], data['mask'], s)

        # Calculate Distortion loss / Rate loss
        losses_dict = self.calculate_loss(output, s)   
        self.manual_backward(losses_dict["loss"])
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer.step()

        # Calculate Aux loss
        aux_loss = self.holo_compression.aux_loss()
        self.manual_backward(aux_loss)
        aux_optimizer.step()
        
        # Calculate Metrics
        metrics_dict = self.calculate_metrics(output)

        # Log
        current_step = self.trainer.global_step * self.trainer.num_devices
        train_loss_dict = {}
        if config.generation or not self.gain:
            for k, v in (losses_dict | metrics_dict).items():
                train_loss_dict[f"{k}_train"] = v
        else:
            for k, v in (losses_dict | metrics_dict).items():
                train_loss_dict[f"{k}_train_level_{s}"] = v
        self.log_dict(train_loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)        

        if current_step % 3200 == 0:
            wandb_logger.log_image(
                key="Training Samples", 
                images=[output["output_poh"], output["output_amp"], output["target_amp"]], 
                caption=["Phase-only Hologram", "Reconstruction", "Label"],
                step=current_step
            )
        return losses_dict["loss"]
    

    def validation_step(self, data, data_idx):
        valid_loss_dict = {}

        # Log
        if config.generation or not self.gain:
            output = self.run_step(data['target'], data['mask'])

            # Calculate Loss
            losses_dict = self.calculate_loss(output)

            # Calculate Metrics
            metrics_dict = self.calculate_metrics(output)

            for k, v in (losses_dict | metrics_dict).items():
                if config.single_lmbda > 0:
                    valid_loss_dict[f"{k}_valid_lmbda_{config.single_lmbda}"] = v
                else: 
                    valid_loss_dict[f"{k}_valid"] = v

            self.log_dict(valid_loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        else:
            total_loss = 0
            for s in range(0, self.holo_compression.levels):
                output = self.run_step(data['target'], data['mask'], s)

                # Calculate Loss        
                losses_dict = self.calculate_loss(output, s)
                total_loss += losses_dict["loss"]

                # Calculate Metrics
                metrics_dict = self.calculate_metrics(output)
            
                for k, v in (losses_dict | metrics_dict).items():
                    valid_loss_dict[f"{k}_valid_level_{s}"] = v
            losses_dict["loss"] = total_loss
            valid_loss_dict[f"total_loss_valid"] = total_loss
            
            self.log_dict(valid_loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
            
        return losses_dict["loss"]


    # Calculate loss
    def calculate_loss(self, output, s=0):
        losses_dict = {}

        output_amp = output["output_amp"]
        target_amp = output["target_amp"]
        output_amp = crop_image(output_amp, self.homography_res, True, stacked_complex=False)
        target_cropped_amp = crop_image(target_amp, self.homography_res, True, stacked_complex=False)

        # Distortion Loss (Scale)
        losses_dict["loss_mse"] = self.criterionMSE(output_amp, target_cropped_amp)

        if not self.image_compression:
            losses_dict["loss"] = losses_dict["loss_mse"]
            return losses_dict
        
        else:
            # Rate Loss
            N, C, H, W = target_amp.size()
            num_pixels = N * C * H * W

            losses_dict["loss_bpp"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
            if self.gain:
                losses_dict["loss"] = self.holo_compression.lmbda[s] * 255 ** 2 * losses_dict["loss_mse"] + losses_dict["loss_bpp"]
            else:
                losses_dict["loss"] = (config.single_lmbda / 10000) * 255 ** 2 * losses_dict["loss_mse"] + losses_dict["loss_bpp"]
            return losses_dict


    # Calculate metrics (PSNR, SSIM)
    def calculate_metrics(self, output):
        output_amp = output["output_amp"]
        target_amp = output["target_amp"]
        output_amp = crop_image(output_amp, self.homography_res, True, False)
        target_cropped_amp = crop_image(target_amp, self.homography_res, True, False)
        psnr = self.psnr(output_amp, target_cropped_amp)
        ssim = self.ssim(output_amp, target_cropped_amp)

        if self.is_train or not self.image_compression:
            return {"metrics_PSNR": psnr, "metrics_SSIM": ssim}
        
        else:
            # Calculate bpp
            _, C, H, W = target_amp.size()
            y_strings = output["strings"]["y"]
            z_strings = output["strings"]["z"]

            bpp_y = (len(y_strings[0][0]) * 8) / (C * H * W)
            bpp_z = (len(z_strings[1][0]) * 8) / (C * H * W)
            bpp = bpp_y + bpp_z
            return {"metrics_PSNR": psnr, "metrics_SSIM": ssim, "metrics_bpp": bpp}


    def calculate_actual_metrics(self, output, filename, idx, s=0):
        output_poh = output["output_poh"]
        input_resolution = output_poh.size()[-2:]
        #conv_size = (2160, 3840)
        #u_in = pad_image(output_poh, conv_size, padval=0, stacked_complex=False)
        phase_out_8bit = phasemap_8bit(output_poh.cpu().detach(), inverted=False, model_name = config.model_name, image_res = config.image_res)
        target_path, target_filename = os.path.split(filename[0])
        target_idx = target_filename.split('_')[-1]
        if s == 0:
            cv2.imwrite(os.path.join('./phases', config.channel, f'{target_idx}.png'), phase_out_8bit)
            # cv2.imwrite(os.path.join('./phases', config.channel, f'{filename}_{idx}.png'), phase_out_8bit) # erase
        else:
            cv2.imwrite(os.path.join('./phases', config.channel, f'qp_{s}', f'{target_idx}.png'), phase_out_8bit)

        output_amp = output["output_amp"]
        target_amp = output["target_amp"]
        output_amp = crop_image(output_amp, self.homography_res, True, False)
        target_cropped_amp = crop_image(target_amp, self.homography_res, True, False)

        psnr = self.psnr(output_amp, target_cropped_amp)
        ssim = self.ssim(output_amp, target_cropped_amp)

        if self.is_train or not self.image_compression:
            return {"metrics_PSNR": psnr, "metrics_SSIM": ssim}
        
        else:
            # Calculate bpp
            _, C, H, W = target_amp.size()
            strings = output["strings"]

            bpp_y = (len(strings[0][0]) * 8) / (C * H * W)
            bpp_z = (len(strings[1][0]) * 8) / (C * H * W)
            bpp = bpp_y + bpp_z
            return {"metrics_PSNR": psnr, "metrics_SSIM": ssim, "metrics_bpp": bpp}
        

    # Test
    def test_step(self, data, data_idx):
        test_loss_dict = {}
        # Log
        if config.generation or not self.gain:
            # data['target'] = data['images']          # erase this
            output = self.run_step(data['target'], data['mask'])

            # sequence_name = data['sequence_name'][0] # erase this
            # data['filename'] = sequence_name         # erase this
            target_path, target_filename = os.path.split(data['filename'][0])
            target_idx = target_filename.split('_')[-1]
            
            target_idx = (data_idx % config.maxIndex) + 1
            
            # Calculate Metrics
            metrics_dict = self.calculate_actual_metrics(output, data['filename'], s=0)
            # metrics_dict = self.calculate_actual_metrics(output, data['filename'], target_idx, s=0) # erase this

            for k, v in (metrics_dict).items():
                if config.single_lmbda > 0:
                    test_loss_dict[f"{k}_test_lmbda_{config.single_lmbda}"] = v
                else:
                    test_loss_dict[f"{k}_test"] = v
                    # logfile.write(f"\nSequence Name:{target_idx} {k}: {v}\n")

            self.log_dict(test_loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        else:
            for s in range(0, self.holo_compression.levels):
                output = self.run_step(data['target'], data['mask'], s)

                target_path, target_filename = os.path.split(data['filename'][0])
                target_idx = target_filename.split('_')[-1]
                
                # Calculate Metrics
                metrics_dict = self.calculate_actual_metrics(output, data['filename'], s=s+1)
            
                for k, v in (metrics_dict).items():
                    test_loss_dict[f"{k}_test_level_{s+1}"] = v
                    # logfile.write(f"\nSequence Name:{target_idx} Level{s:.3f} {k}: {v}\n")
            
            self.log_dict(test_loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        # logfile.close()

    # Numerically Reconstruct Phase-only Hologram
    def reconstruction(self, final_phase):
        final_complex = torch.exp(1j * final_phase)

        # Propagation SLM to Target
        if self.fullcolor:
            r_final, g_final, b_final = torch.chunk(final_complex, 3, dim=1)
            r_slm = propagation(r_final, self.featuresize, self.wavelength['r'], self.propdist)
            g_slm = propagation(g_final, self.featuresize, self.wavelength['g'], self.propdist)
            b_slm = propagation(b_final, self.featuresize, self.wavelength['b'], self.propdist)
            slm = torch.cat((r_slm, g_slm, b_slm), dim=1)

        else:
            slm = propagation(final_complex, self.featuresize, self.wavelength[self.channel], self.propdist)

        final_amp = torch.abs(slm).float()
        return final_amp
    

if __name__ == "__main__":
    # Set Configuration
    config = Config()
    seed_everything(config.seed, workers=True)

    train_mode = 'train' if config.train else 'test'
    logger_name = f'{train_mode}_{config.stage_mode}_{config.channel}'
    if config.image_compression:
        logger_name += f"-image-compression-{config.mode}"
        if config.single_lmbda > 0:
            logger_name += f"-lmbda-{config.single_lmbda}"

    # Load Data
    data_module = LightningDataModule(config)
    train_data_loader = data_module.train_dataloader()
    valid_data_loader = data_module.val_dataloader()
    test_data_loader = data_module.test_dataloader()


    # Model Checkpoint
    if config.mode:
        model_dir = f'./experiment/{config.model_name}/{config.channel}/{config.stage_mode}-{config.mode}'
    else:
        model_dir = f'./experiment/{config.model_name}/{config.channel}/{config.stage_mode}'
    if config.single_lmbda > 0:
        model_dir += f"/lmbda_{config.single_lmbda}"
    os.makedirs(model_dir, exist_ok=True)


    rich_progressbar = RICH(leave=True)
    wandb_logger = WandbLogger(name=logger_name, project=config.model_name)


    if config.generation:
        model_checkpoint_PSNR = ModelCheckpoint(
            dirpath=model_dir, 
            filename="best-psnr-{epoch}-{metrics_PSNR_valid:.3f}", 
            monitor="metrics_PSNR_valid", 
            mode="max", 
            every_n_epochs=1, 
            save_top_k=1
        )

        model_checkpoint_latest = ModelCheckpoint(
            dirpath=model_dir, 
            filename="latest-{epoch}-{step}", 
            monitor="step", 
            mode="max", 
            every_n_epochs=1, 
            save_top_k=1
        )
        callbacks = [rich_progressbar, model_checkpoint_PSNR, model_checkpoint_latest]

    elif config.image_compression and config.gain:
        model_checkpoint_Joint_Loss = ModelCheckpoint(
            dirpath=model_dir, 
            filename="best-loss-{epoch}", 
            monitor="total_loss_valid", 
            mode="min", 
            every_n_epochs=1, 
            save_top_k=1
        )

        model_checkpoint_latest = ModelCheckpoint(
            dirpath=model_dir, 
            filename="latest-{epoch}-{step}", 
            monitor="step", 
            mode="max", 
            every_n_epochs=1, 
            save_top_k=1
        )

        callbacks = [rich_progressbar, model_checkpoint_Joint_Loss, model_checkpoint_latest]

    elif config.image_compression and not config.gain:
        model_checkpoint_Joint_Loss = ModelCheckpoint(
            dirpath=model_dir, 
            filename="best-loss-{epoch}", 
            monitor=f"loss_valid_lmbda_{config.single_lmbda}", 
            mode="min", 
            every_n_epochs=1, 
            save_top_k=1
        )
        
        model_checkpoint_latest = ModelCheckpoint(
            dirpath=model_dir, 
            filename="latest-{epoch}-{step}", 
            monitor="step", 
            mode="max", 
            every_n_epochs=1, 
            save_top_k=1
        )

        callbacks = [rich_progressbar, model_checkpoint_Joint_Loss, model_checkpoint_latest]


    # Trainer
    # strategy = DDPStrategy(process_group_backend="gloo")    # Train or Test in Windows setting
    strategy = 'auto' if config.image_compression else 'ddp_find_unused_parameters_true'
    trainer = Trainer(
        accelerator=config.device,      
        devices=config.gpu,         # the number of devices
        deterministic="warn",       # for reproducibility
        benchmark=False,            # for reproducibility
        strategy=strategy,

        logger=wandb_logger,        # Log on wandb   
        log_every_n_steps=1,
        callbacks=callbacks,
        
        max_epochs=config.epoches,  # Training epochs
        check_val_every_n_epoch=1,  # Do validation per one epoch
    )


    # Train
    if config.train:
        model = ImageCompressionModule(config)
        if config.generation:
            if config.load_model:
                model_ckpts = os.listdir(f'./experiment/{config.model_name}/{config.channel}/{config.stage_mode}')
                model_ckpt = [os.path.join(f'./experiment/{config.model_name}/{config.channel}/{config.stage_mode}', model_ckpt) for model_ckpt in model_ckpts if 'latest' in model_ckpt][0]
                model = ImageCompressionModule.load_from_checkpoint(
                    checkpoint_path=model_ckpt, 
                    config=config
                    )
        elif config.image_compression:
            if config.load_model:
                model_ckpts = os.listdir(f'./experiment/{config.model_name}/{config.channel}/stage2_lmbda_10')
                model_ckpt = [os.path.join(f'./experiment/{config.model_name}/{config.channel}/stage2_lmbda_10', model_ckpt) for model_ckpt in model_ckpts if 'loss' in model_ckpt][0]
                model = ImageCompressionModule.load_from_checkpoint(
                    checkpoint_path=model_ckpt, 
                    config=config
                    )
            else:
                model_ckpts = os.listdir(f'./experiment/{config.model_name}/{config.channel}/stage1')
                model_ckpt = [os.path.join(f'./experiment/{config.model_name}/{config.channel}/stage1', model_ckpt) for model_ckpt in model_ckpts if 'psnr' in model_ckpt][0]
                model = ImageCompressionModule.load_from_checkpoint(
                    checkpoint_path=model_ckpt, 
                    config=config
                    )
        
        
        trainer.fit(model=model, train_dataloaders=train_data_loader, val_dataloaders=valid_data_loader)

    # Test
    elif not config.train:
        config.gpu = config.gpu[0]
        model_ckpts = os.listdir(model_dir)
        if config.stage_mode == 'stage1':
            model_ckpt = [os.path.join(model_dir, model_ckpt) for model_ckpt in model_ckpts if 'psnr' in model_ckpt][0]
        else:
            model_ckpt = [os.path.join(model_dir, model_ckpt) for model_ckpt in model_ckpts if 'loss' in model_ckpt][0]
        model = ImageCompressionModule.load_from_checkpoint(
            checkpoint_path=model_ckpt, 
            map_location=f'cuda:{config.gpu}',
            config=config
        )
        model.holo_compression.update()
        trainer.test(model=model, dataloaders=test_data_loader, verbose=True)
        