import os
import math
import random
import cv2

import torch
from torch import nn, optim
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar as RICH
from pytorch_lightning.loggers import WandbLogger
# from lightning.pytorch.strategies import DDPStrategy

from config import Config
from nhvc.joint_compression import SpatioTemporalHyperpriorCoder
from set_lightning_datamodule import LightningDataModule
from utils import phasemap_8bit
from propagation import propagation
from train import ImageCompressionModule


# Setting up the environment variable for Weights & Biases API Key
os.environ["WANDB_API_KEY"] = ""
torch.set_float32_matmul_precision('medium')

class VideoCompressionModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False
        self.configure_properties(config)
        self.holo_image_compression = ImageCompressionModule.load_from_checkpoint(checkpoint_path=config.model_ckpt)
        self.holo_image_compression.requires_grad_(False)
        self.holo_video_compression = SpatioTemporalHyperpriorCoder(gain=config.gain)
        self.configure_metrics()
        self.save_hyperparameters()
    
    def configure_properties(self, config):
        self.is_train = config.train
        cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
        self.channel = config.channel
        self.fullcolor = self.channel == 'n'
        self.wavelength = {'b': 450 * nm, 'g': 520 * nm, 'r': 638 * nm}
        self.featuresize = (config.featuresize * um, config.featuresize * um)
        self.propdist = config.propdist * cm
        self.image_res = config.image_res
        self.homography_res = config.homography_res
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.aux_lr = config.aux_lr
        self.lr_step_size = config.lr_step_size
        self.criterionMSE = nn.MSELoss()
        self.filename = "readysteady.txt"
        
    def configure_metrics(self):
        self.psnr = PSNR(data_range=1.0, reduction='elementwise_mean')
        self.ssim = SSIM(gaussian_kernel=True, sigma=1.5, kernel_size=11, reduction='elementwise_mean', data_range=1.0, k1=0.01, k2=0.03)
    
    # Set optimizer
    def configure_optimizers(self):
        model = self.holo_video_compression

        parameters = {n for n, p in model.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
        aux_parameters = {n for n, p in model.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
        params_dict = {n: p for n, p in model.named_parameters()}

        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters
        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0
        
        optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=self.lr)
        aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=self.aux_lr)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
        return [optimizer, aux_optimizer], [lr_scheduler]
    

    # Stage 3 
    def stage_3(self, current, reference, s=0):
        if self.is_train:
            output = self.holo_video_compression(current, reference.detach(), s)
        # For Testing
        else:
            compressed = self.holo_video_compression.compress(current, reference, s)
            decompressed = self.holo_video_compression.decompress(compressed["strings"], compressed["shape"], reference, s)
            output = compressed | decompressed
            output["output_poh"] = self.holo_image_compression.holo_compression.getX(output["y_hat"], s)["output_poh"]
            
        return output
    
    def get_latent(self, target, s=0):
        slm_amp_phase = self.holo_image_compression.initial_hologram_generation(target)
        return self.holo_image_compression.holo_compression.getY(slm_amp_phase, s)
    
    def stage_intra(self, target, mask, s):
        slm_amp_phase = self.holo_image_compression.initial_hologram_generation(target)
        compressed = self.holo_image_compression.holo_compression.compress(slm_amp_phase, s)
        decompressed = self.holo_image_compression.holo_compression.decompress(compressed, s)
        output = compressed | decompressed
        output["target_amp"] = target * mask
        return output

    #############################
    #       Training Step       #
    #############################
    def training_step(self, data, data_idx):
        global y_ref
        self.holo_image_compression.train()
        images = [img for img in data['images']]
        optimizer, aux_optimizer = self.optimizers()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        s = random.randint(0, self.holo_image_compression.holo_compression.levels - 1) if config.gain else 0
        
        for idx in range(len(images)):
            if idx == 0:
                latent = self.get_latent(images[idx], s)
                y_ref = latent["gained_y_hat"]
            else:
                optimizer.zero_grad()
                aux_optimizer.zero_grad()
                latent = self.get_latent(images[idx], s)
                y_cur = latent["gained_y"]

                output = self.stage_3(y_cur, y_ref, s)
                y_ref = output['y_hat']

                # Calculate Distortion loss / Rate loss
                losses_dict = self.calculate_loss(output, images[idx])   
                self.manual_backward(losses_dict["loss"])
                self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                optimizer.step()

                # Calculate Aux loss
                aux_loss = self.holo_video_compression.aux_loss()
                self.manual_backward(aux_loss)
                aux_optimizer.step()

                # Log
                current_step = self.trainer.global_step * self.trainer.num_devices
                train_loss_dict = {}
                for k, v in (losses_dict).items():
                    train_loss_dict[f"{k}_train_level_{s}"] = v
                self.log_dict(train_loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)        

        return losses_dict["loss"]
    

    def validation_step(self, data, data_idx):
        valid_loss_dict = {}
        # Log
        self.holo_image_compression.eval()
        total_loss = 0
        images = [img for img in data['images']]
        if not config.gain:
            for idx in range(len(images)):
                if idx == 0:
                    latent = self.get_latent(images[idx])
                    y_ref = latent["gained_y_hat"]
                else:
                    latent = self.get_latent(images[idx])
                    y_cur = latent["gained_y"]
                    output = self.stage_3(y_cur, y_ref)
                    y_ref = output['y_hat']

                    # Calculate Loss        
                    losses_dict = self.calculate_loss(output, images[idx])   
                    total_loss += losses_dict["loss"]

        else:
            for s in range(0, self.holo_image_compression.holo_compression.levels):
                level_loss = 0
                for idx in range(len(images)):
                    if idx == 0:
                        latent = self.get_latent(images[idx], s)
                        y_ref = latent["gained_y_hat"]
                    else:
                        latent = self.get_latent(images[idx], s)
                        y_cur = latent["gained_y"]
                        output = self.stage_3(y_cur, y_ref, s)
                        y_ref = output['y_hat']

                        # Calculate Loss        
                        losses_dict = self.calculate_loss(output, images[idx])   
                        level_loss += losses_dict["loss"]

                total_loss += level_loss
                valid_loss_dict[f"loss_valid_level_{s}"] = level_loss
            
        losses_dict["loss"] = total_loss
        valid_loss_dict[f"total_loss_valid"] = total_loss
        self.log_dict(valid_loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        self.holo_image_compression.train()

        return losses_dict["loss"]


    # Calculate loss
    def calculate_loss(self, output, target):
        losses_dict = {}

        # Rate Loss
        N, C, H, W = target.size()
        num_pixels = N * C * H * W

        losses_dict["loss_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        losses_dict["loss"] = losses_dict["loss_bpp"]
        return losses_dict


    # Calculate metrics (PSNR, SSIM)
    def calculate_actual_metrics(self, output, seq_name, idx, s=0):
        output_poh = output["output_poh"]
        phase_out_8bit = phasemap_8bit(output_poh.cpu().detach(), inverted=False)
        frame_idx = (idx % config.maxIndex) + 1

        if s == 0:
            cv2.imwrite(os.path.join('./phases', config.channel, f'{seq_name}_{frame_idx}.png'), phase_out_8bit)
        else:
            cv2.imwrite(os.path.join('./phases', config.channel, f'qp_{s}', f'{seq_name}_{frame_idx}.png'), phase_out_8bit)


        # output_amp = output["output_amp"]
        target_amp = output["target_amp"]
        
        logfile = open(self.filename, 'a+')
        
        # Calculate bpp
        _, C, H, W = target_amp.size()
        strings = output["strings"]
        bpp_y = (len(strings[0][0]) * 8) / (C * H * W)
        bpp_z = (len(strings[1][0]) * 8) / (C * H * W)
        bpp = bpp_y + bpp_z
        
        logfile.write(f"\nSequence Name:{frame_idx}  Channel:{config.channel}  BPP: {bpp}\n")
        logfile.close()
        
        return {"metrics_bpp": bpp}


    # Test
    def test_step(self, data, data_idx):
        test_loss_dict = {}
        global y_ref
        self.holo_image_compression.eval()
        
        s = config.level

        images = data['images']
        sequence_name = data['sequence_name'][0]

        if config.allIntra:
            output = self.stage_intra(images, data['mask'], s)
            y_ref = output["gained_y_hat"]
            metrics_dict = self.calculate_actual_metrics(output, sequence_name, data_idx, s=s)
            for k, v in (metrics_dict).items():
                test_loss_dict[f"{k}_test_allIntra_level_{s}"] = v

        else:
            if data['intra_frame']:
                output = self.stage_intra(images, data['mask'], s)
                y_ref = output["gained_y_hat"]
                metrics_dict = self.calculate_actual_metrics(output, sequence_name, data_idx, s=s+1)
                for k, v in (metrics_dict).items():
                    test_loss_dict[f"{k}_test_level_{s+1}"] = v
            else:
                latent = self.get_latent(images, s)
                y_cur = latent["gained_y"]
                output = self.stage_3(y_cur, y_ref, s)
                y_ref = output['y_hat']
                output["target_amp"] = images

                # Calculate Loss        
                metrics_dict = self.calculate_actual_metrics(output, sequence_name, data_idx, s=s+1)
                for k, v in (metrics_dict).items():
                    test_loss_dict[f"{k}_test_level_{s+1}"] = v
            
        self.log_dict(test_loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

    # Numerically Reconstruct Phase-only Hologram
    def reconstruction(self, final_phase):
        final_complex = torch.exp(1j * final_phase)
        slm = propagation(final_complex, self.featuresize, self.wavelength[self.channel], self.propdist)
        final_amp = torch.abs(slm).float()
        return final_amp
    

if __name__ == "__main__":
    # Set Configuration
    config = Config()
    seed_everything(config.seed, workers=True)

    # Logger
    if config.generation:
        stage_mode = 'stage1'
    elif config.image_compression:
        stage_mode = 'stage2'
    else:
        stage_mode = 'stage3'

    train_mode = 'train' if config.train else 'valid'
    logger_name = f'{train_mode}_{stage_mode}_{config.channel}'
    if config.video_compression:
        logger_name += f"-video-compression-{config.mode}"
        if config.single_lmbda > 0:
            logger_name += f"-lmbda-{config.single_lmbda}"

    rich_progressbar = RICH(leave=True)
    wandb_logger = WandbLogger(name=logger_name, project=config.model_name)

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
    # os.makedirs(f'{model_dir}/test', exist_ok=True)

    model_checkpoint_Loss = ModelCheckpoint(
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
    callbacks = [rich_progressbar, model_checkpoint_Loss, model_checkpoint_latest]


    # Trainer
    # strategy = DDPStrategy(process_group_backend="gloo")    # Train or Test in Windows setting
    strategy = 'auto' if config.image_compression or config.video_compression else 'ddp_find_unused_parameters_true'
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
        inter_model = VideoCompressionModule(config)
        if config.load_model:
            model_ckpts = os.listdir(f'./experiment/{config.model_name}/stage3')
            model_ckpt = [os.path.join(f'./experiment/{config.model_name}/stage3', model_ckpt) for model_ckpt in model_ckpts if 'loss' in model_ckpt][0]
            print(model_ckpt)
            inter_model = VideoCompressionModule.load_from_checkpoint(
                checkpoint_path=model_ckpt, 
                config=config
                )
            
        trainer.fit(model=inter_model, train_dataloaders=train_data_loader, val_dataloaders=valid_data_loader)

    # Test
    elif not config.train:
        config.gpu = config.gpu[0]
        model_ckpts = os.listdir(model_dir)
        print(model_ckpts)
        model_ckpt = [os.path.join(model_dir, model_ckpt) for model_ckpt in model_ckpts if 'loss' in model_ckpt][0]
        print(model_ckpt)
        
        model = VideoCompressionModule.load_from_checkpoint(
            checkpoint_path=model_ckpt, 
            map_location=f'cuda:{config.gpu}',
            config=config
        )
        

        # summary(model.holo_video_compression, [(1, 1088, 1920), (1, 1088, 1920), (1,)])
        model.holo_image_compression.holo_compression.update()
        model.holo_video_compression.update()
        
        # model = model.holo_image_compression.initial_phase_generation

        # def count_parameters(model):
        #     return sum(p.numel() for p in model.parameters())
        # # Print total parameters for the entire model
        # total_params = count_parameters(model)
        # print(f"Total number of parameters in the entire model: {total_params}")

        # # Print parameters for each specified module
        # modules = ['net']
        # for module_name in modules:
        #     module = getattr(model, module_name)
        #     num_params = count_parameters(module)
        #     print(f"Total number of parameters in {module_name}: {num_params}")
        # # from torchsummary import summary as summary
        
        trainer.test(model=model, dataloaders=test_data_loader, verbose=True)
        