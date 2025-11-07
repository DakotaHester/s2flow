import logging
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchmetrics.functional as TMF
from tqdm import tqdm
from ..metrics import MultispectralLPIPS
from ..utils import get_device, get_hp_dtype

logger = logging.getLogger(__name__)

class FlowMatchingSRTrainer:
    
    def __init__(self, config: Dict[str, Any], model: nn.Module):
        
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        
        # model output directory
        self.out_dir = Path(config.get('job', {}).get('out_dir', './runs'))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.out_dir / config.get('job', {}).get('checkpoint_filename', 'checkpoint.pt')
        self.log_dir = Path(config.get('job', {}).get('logging', {}).get('log_dir', './logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        hyperparameters = config.get('hyperparameters', None)
        if hyperparameters is None:
            raise ValueError("Hyperparameters must be specified in the config under 'hyperparameters'")
        self.init_learning_rate = hyperparameters.get('learning_rate', 1e-3)
        self.batch_size = hyperparameters.get('batch_size', 1024)
        self.micro_batch_size = hyperparameters.get('micro_batch_size', 32)
        self.num_epochs = hyperparameters.get('num_epochs', 300)
        self.warmup_epochs = hyperparameters.get('warmup_epochs', 10)
        self.weight_decay = hyperparameters.get('weight_decay', 1e-2)
        self.use_amp = hyperparameters.get('use_amp', True)
        
        self.grad_accum_steps = self.batch_size // self.micro_batch_size
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.init_learning_rate,
            weight_decay=hyperparameters.get('weight_decay', 1e-2),
        )
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs - 1,
        )
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs - self.warmup_epochs,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[self.warmup_epochs],
        )
        self.current_epoch = 1
        
        self.hp_dtype = torch.float32
        if self.use_amp:
            self.hp_dtype = get_hp_dtype()
        
        if self.hp_dtype != torch.float32:
            self.scaler = GradScaler(enabled=True)
        else:
            self.scaler = None
        
        self.lpips_metric = MultispectralLPIPS(config)
        
        self.metrics = {
            'epoch': [],
            'lr': [],
        }
        self.metric_names = ['l1_loss', 'psnr', 'ssim', 'mssim', 'lpips']
        for phase in ('train', 'val'):
            for metric in self.metric_names:
                self.metrics[f'{phase}_{metric}'] = []
                
        logger.info(f"Initialized {self.__class__.__name__} on device {self.device} with model {type(self.model).__name__}")


    @classmethod
    def from_checkpoint(cls, config: Dict[str, Any], model: nn.Module):
        
        trainer = cls(config, model)
        trainer.load_checkpoint()
        return trainer
    
    
    def fit(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader]=None) -> None:
        
        for epoch in range(self.current_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            
            self.metrics['epoch'].append(epoch)
            self.metrics['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            logger.debug(f"Starting epoch {epoch}/{self.num_epochs}...")
            epoch_metrics = self._run_epoch(train_dataloader, val_dataloader)
            
            for metric in self.metric_names:
                for phase in ('train', 'val'):
                    self.metrics[f'{phase}_{metric}'].append(epoch_metrics.get(f'{phase}_{metric}', None))
            
            self.lr_scheduler.step()
            self.save_checkpoint()
            self.save_model()
            self.save_metrics()
    
    
    def _run_epoch(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader]=None) -> None:
        
        if val_dataloader is not None:
            phases = ('train', 'val')
            logger.debug("Validation dataloader provided; running both training and validation phases.")
        else:
            phases = ('train',)
            logger.debug("No validation dataloader provided; running only training phase.")
        
        epoch_metrics = {
            f'{phase}_{metric}': 0.0
            for metric in self.metric_names
            for phase in phases
        }
        
        for phase in phases:
            torch.set_grad_enabled(phase == 'train')
            if phase == 'train':
                self.model.train()
                dataloader = train_dataloader
            else:
                if val_dataloader is None:
                    continue
                self.model.eval()
                dataloader = val_dataloader
            
            phase_count = 0
            self.optimizer.zero_grad(set_to_none=True)
            
            with tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {self.current_epoch}/{self.num_epochs}", unit="batches") as pbar:
                for batch_num, batch_data in enumerate(pbar):
                    
                    step_metrics = self._step(batch_num, batch_data, phase)
                    phase_count += batch_data[0].size(0)
                    
                    for k, v in step_metrics.items():
                        epoch_metrics[k] += v
                    
                    metrics_fmt = {'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'}
                    for metric in self.metric_names:
                        avg = epoch_metrics[f'{phase}_{metric}'] / phase_count
                        metrics_fmt[metric] = f'{avg:.2e}'
                    pbar.set_postfix(metrics_fmt)
                    logger.debug(f'Epoch {self.current_epoch} {phase} step {batch_num} metrics:\n', '\n'.join([f'{k}: {v}' for k, v in metrics_fmt.items()]))

            for metric in self.metric_names:
                epoch_metrics[f'{phase}_{metric}'] /= phase_count
            
        return epoch_metrics

      
    def _step(self, batch_num: int, batch_data: Tuple[torch.Tensor], phase: str) -> Dict[str, float]:
        
        input_img, target_img = batch_data
        input_img, target_img = input_img.to(self.device), target_img.to(self.device)
        
        x_0 = torch.randn_like(target_img)
        t = torch.rand(input_img.size(0), device=self.device)
        t_batch = t.view(-1, 1, 1, 1)
        
        # linear interpolation between input and noise to form optimal transport map
        x_t = (1 - t_batch) * x_0 + t_batch * input_img
        target_vector = input_img - x_0
        model_input = torch.cat([x_t, input_img], dim=1)
        
        with autocast(device_type=self.device.type, dtype=self.hp_dtype, enabled=self.use_amp):
            pred_vector = self.model(model_input, t)
            loss = F.l1_loss(pred_vector, target_vector, reduction='none').mean(dim=(1, 2, 3))
        
        if phase == 'train':
            total_loss = (loss / self.grad_accum_steps).mean()
            
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            if (batch_num + 1) % self.grad_accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        
        pred_image = (pred_vector + x_0).detach().clamp(-1.0, 1.0)
        return {
            f'{phase}_l1_loss': loss.detach().sum().item(),
            f'{phase}_psnr': TMF.image.peak_signal_noise_ratio(pred_image, target_img, data_range=(-1, 1), reduction='none', dim=(1, 2, 3)).sum().item(),
            f'{phase}_ssim': TMF.image.structural_similarity_index_measure(pred_image, target_img, data_range=(-1, 1), reduction='none').sum().item(),
            f'{phase}_mssim': TMF.image.multiscale_structural_similarity_index_measure(pred_image, target_img, data_range=(-1, 1), reduction='none').sum().item(),
            f'{phase}_lpips': self.lpips_metric.forward(pred_image, target_img, reduction='none').sum().item(),
        }
    

    def save_checkpoint(self):
        
        logger.debug(f"Saving checkpoint to {self.checkpoint_path} at epoch {self.current_epoch}...")
        
        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'metrics': self.metrics,
        }
        torch.save(checkpoint_dict, self.checkpoint_path)
        logger.debug(f"Checkpoint saved to {self.checkpoint_path}.")
    
    
    def load_checkpoint(self):
        
        logger.debug(f"Loading checkpoint from {self.checkpoint_path}...")
        
        if not self.checkpoint_path.exists():
            logger.warning(f"Checkpoint file {self.checkpoint_path} does not exist. Starting from scratch.")
            return
        
        checkpoint_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])
        self.current_epoch = checkpoint_dict['current_epoch'] + 1
        self.metrics = checkpoint_dict['metrics']
        
        logger.info(f"Loaded checkpoint from {self.checkpoint_path}, resuming from epoch {self.current_epoch}.")
    
    
    def save_metrics(self):
        log_file = self.log_dir / 'training_metrics.csv'
        logger.debug(f"Saving training metrics to {log_file}...")
        
        pd.DataFrame(self.metrics).to_csv(log_file, index=False)
    
    def save_model(self):
        model_file = self.out_dir / 'model.pt'
        logger.debug(f"Saving model to {model_file}...")
        torch.save(self.model.state_dict(), model_file)
        logger.debug(f"Model saved to {model_file}.")


class LandCoverTrainer:
    ...
    
def train_sr_model(config: Dict[str, Any], model: nn.Module):
    
    load_checkpoint = config.get('job', {}).get('load_checkpoint', False)
    if load_checkpoint:
        logger.info("`load_checkpoint` is True; loading trainer from checkpoint...")
        trainer = FlowMatchingSRTrainer.from_checkpoint(config, model)
    else:
        logger.info("`load_checkpoint` is False; initializing new trainer...")
        trainer = FlowMatchingSRTrainer(config, model)
    
    logger.info("Starting super-resolution model training...")
    trainer.train(config)
    logger.info("Super-resolution model training complete.")
    


def train_lc_model(config: Dict[str, Any]):
    trainer = LandCoverTrainer()
    logger.info("Starting land cover model training...")
    trainer.train(config)