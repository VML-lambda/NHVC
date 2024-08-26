import os
from copy import deepcopy

import torch
from torch import Tensor
from lightning.pytorch.callbacks import Callback


class BestPSNRSaveCallback(Callback):
    def __init__(self, model_dir, monitor='psnr', save_weights_only=False):
        super().__init__()
        self.model_dir = f'{model_dir}/best-metrics-PSNR/'
        os.makedirs(self.model_dir, exist_ok=True)

        self.monitor = monitor
        self.save_weights_only = save_weights_only
        self.best_psnr = 0.0

    def on_validation_end(self, trainer):
        monitor_candidates = deepcopy(trainer.callback_metrics)
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)

        current_psnr = monitor_candidates.get(self.monitor)
        if current_psnr > self.best_psnr:
            self.best_psnr = current_psnr
            print(f'Epoch: {trainer.current_epoch}, \tStep:{trainer.global_step}, \tBest PSNR: {self.best_psnr}')
            trainer.save_checkpoint(
                f'{self.model_dir}/best_model_checkpoint.ckpt', 
                weights_only=self.save_weights_only
            )


class BestSSIMSaveCallback(Callback):
    def __init__(self, model_dir, monitor='ssim', save_weights_only=False):
        super().__init__()
        self.model_dir = f'{model_dir}/best-metrics-SSIM/'
        os.makedirs(self.model_dir, exist_ok=True)

        self.monitor = monitor
        self.save_weights_only = save_weights_only
        self.best_ssim = 0.0

    def on_validation_end(self, trainer):
        monitor_candidates = deepcopy(trainer.callback_metrics)
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)

        current_ssim = monitor_candidates.get(self.monitor)
        if current_ssim > self.best_ssim:
            self.best_ssim = current_ssim
            print(f'Epoch: {trainer.current_epoch}, \tStep:{trainer.global_step}, \tBest SSIM: {self.best_ssim}')
            trainer.save_checkpoint(
                f'{self.model_dir}/best_model_checkpoint.ckpt', 
                weights_only=self.save_weights_only
            )


class BestLossSaveCallback(Callback):
    def __init__(self, model_dir, monitor='loss', save_weights_only=False):
        super().__init__()
        self.model_dir = f'{model_dir}/best-metrics-SSIM/'
        os.makedirs(self.model_dir, exist_ok=True)

        self.monitor = monitor
        self.save_weights_only = save_weights_only
        self.best_loss = 65332.0  # 가장 낮은 Loss 초기값

    def on_validation_end(self, trainer):
        monitor_candidates = deepcopy(trainer.callback_metrics)
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)

        current_loss = monitor_candidates.get(self.monitor)
        if current_loss > self.best_loss:
            self.best_loss = current_loss
            print(f'Epoch: {trainer.current_epoch}, \tStep:{trainer.global_step}, \tBest Loss: {self.best_loss}')
            trainer.save_checkpoint(
                f'{self.model_dir}/best_model_checkpoint.ckpt', 
                weights_only=self.save_weights_only
            )