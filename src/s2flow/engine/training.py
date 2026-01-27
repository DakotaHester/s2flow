from collections import defaultdict
from collections.abc import Callable
import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchmetrics.functional as TMF
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm
from functools import partial
from abc import ABC, abstractmethod
from ..loss import focal_loss, MultispectralPerceptualLoss
from ..metrics import MultispectralLPIPS, MetricsTracker
from ..utils import get_device, get_hp_dtype
from ..modules import UNetDiscriminatorSN

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """
    Abstract base class handling training loop and orchestration.
    """
    
    def __init__(self, config: Dict[str, Any], model: nn.Module):
        logger.debug(f"Initializing {self.__class__.__name__}...")
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        
        # Paths
        self.out_path = config['paths']['out_path']
        self.log_path = config['paths']['log_path']
        self.checkpoint_path = self.out_path / config.get('job', {}).get('checkpoint_filename', 'checkpoint.pt')
        self.model_path = self.out_path / 'model.pt'
        self.best_model_path = self.out_path / 'best_model.pt'
        logger.debug(f"Output path: {self.out_path}")
        logger.debug(f"Checkpoint path: {self.checkpoint_path}")
        logger.debug(f"Best model path: {self.best_model_path}")
        logger.debug(f"Log path: {self.log_path}")
                
        # Hyperparameters
        hp = config.get('hyperparameters', {})
        self.init_lr = hp.get('learning_rate', 1e-3)
        self.batch_size = hp.get('batch_size', 1024)
        self.micro_batch_size = hp.get('micro_batch_size', 32)
        self.num_epochs = hp.get('num_epochs', 300)
        self.warmup_epochs = hp.get('warmup_epochs', 10)
        self.use_amp = hp.get('use_amp', True)
        self.grad_accum_steps = self.batch_size // self.micro_batch_size
        
        logger.debug(f"Training config: epochs={self.num_epochs}, batch_size={self.batch_size}, "
                     f"micro_batch={self.micro_batch_size}, accum_steps={self.grad_accum_steps}, amp={self.use_amp}")
        
        # Best Model Monitoring (Defaults)
        self.monitor_config: Optional[Dict[str, str]] = None 
        self.best_metric_value: Optional[float] = None
        self.loss_name = 'loss'

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.init_lr, weight_decay=hp.get('weight_decay', 1e-2)
        )
        self.current_epoch = 1
        self.lr_scheduler = self._create_lr_scheduler()
        
        # AMP
        self.hp_dtype = get_hp_dtype() if self.use_amp else torch.float32
        self.scaler = GradScaler(enabled=True) if (self.use_amp and self.hp_dtype == torch.float16) else None
        if self.scaler:
            logger.debug("GradScaler initialized for float16 mixed precision.")
        
        # Task Specifics
        self._init_task_specific()
        
        # Initialize Tracker
        self.metrics_tracker = MetricsTracker(
            self._get_metric_functions(), 
            loss_name=self.loss_name
        )
        
        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")

    @property
    def metrics(self):
        return self.metrics_tracker.history

    def _create_lr_scheduler(self):
        logger.debug("Creating LR scheduler with warmup...")
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs - self.warmup_epochs
        )
        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs - 1
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
            )
        return cosine
    
    def _init_task_specific(self):
        pass
    
    @abstractmethod
    def _get_metric_functions(self) -> Dict[str, Callable]:
        pass
    
    @abstractmethod
    def _compute_loss_and_predictions(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
    
    @classmethod
    def from_checkpoint(cls, config: Dict[str, Any], model: nn.Module):
        logger.debug("Initializing trainer from checkpoint...")
        trainer = cls(config, model)
        trainer.load_checkpoint()
        return trainer
    
    def fit(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> None:
        logger.info("Starting training loop.")
        for epoch in range(self.current_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            self.metrics_tracker.reset_epoch()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics_tracker.update_epoch(epoch, current_lr)
            
            logger.debug(f"--- Epoch {epoch}/{self.num_epochs} (LR: {current_lr:.2e}) ---")
            
            self._run_phase('train', train_dataloader)
            if val_dataloader:
                self._run_phase('val', val_dataloader)
            
            epoch_results = self.metrics_tracker.finalize_epoch()
            
            if self.monitor_config and val_dataloader:
                self._check_best_model(epoch_results)

            self.lr_scheduler.step()
            self.save_checkpoint()
            self.save_metrics()
            self.save_model()
            
        if self.monitor_config and self.best_model_path.exists():
            logger.info(f"Training finished. Loading best model from {self.best_model_path}...")
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        else:
            logger.info("Training finished.")

    def _run_phase(self, phase: str, dataloader: DataLoader):
        is_train = phase == 'train'
        torch.set_grad_enabled(is_train)
        self.model.train() if is_train else self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        
        logger.debug(f"Starting {phase} phase. Samples: {len(dataloader.dataset)}")
        
        with tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {self.current_epoch}", unit="bt") as pbar:
            for i, batch in enumerate(pbar):
                # Detailed debug for first batch to catch data loading issues immediately
                if i == 0:
                    logger.debug(f"[{phase}] First batch loaded. Input shape (approx): {batch[0].shape}")
                
                loss, preds, targets = self._compute_loss_and_predictions(batch)
                
                if is_train:
                    self._backward_step(loss, i)

                batch_metrics = self.metrics_tracker.update_batch(
                    phase, loss.detach(), preds.detach(), targets
                )
                
                pbar.set_postfix({
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    **{k: f'{v:.2e}' for k, v in batch_metrics.items()}
                })

    def _backward_step(self, loss: torch.Tensor, batch_idx: int):
        scaled_loss = loss.mean() / self.grad_accum_steps
        
        if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
            logger.error(f"NaN/Inf detected in scaled loss at batch {batch_idx}!")
        
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
            
        if (batch_idx + 1) % self.grad_accum_steps == 0:
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                logger.debug(f"Scaler scale factor: {self.scaler.get_scale()}")
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def _check_best_model(self, results: Dict[str, float]):
        metric_name = self.monitor_config['metric']
        mode = self.monitor_config['mode']
        
        if self.best_metric_value is None:
            self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
            logger.debug(f"Initialized best_metric_value to {self.best_metric_value} (mode: {mode})")
            
        current_val = results.get(metric_name) or results.get(f'val_{metric_name}')
        if current_val is None: 
            logger.warning(f"Monitor metric '{metric_name}' not found in results: {list(results.keys())}")
            return

        improved = (current_val < self.best_metric_value) if mode == 'min' else (current_val > self.best_metric_value)
        
        if improved:
            logger.info(f"New best {metric_name}: {current_val:.6f} (prev: {self.best_metric_value:.6f}). Saving model.")
            self.best_metric_value = current_val
            torch.save(self.model.state_dict(), self.best_model_path)
        else:
            logger.debug(f"No improvement in {metric_name}. Current: {current_val:.6f}, Best: {self.best_metric_value:.6f}")

    def save_checkpoint(self):
        logger.debug(f"Saving checkpoint to {self.checkpoint_path}...")
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.current_epoch,
            'metrics': self.metrics_tracker.to_dict(),
            'best_metric_value': self.best_metric_value
        }
        if self.scaler: state['scaler'] = self.scaler.state_dict()
        torch.save(state, self.checkpoint_path)
    
    
    def save_model(self):
        model_path = self.out_path / 'model.pt'
        logger.debug(f"Saving latest model to {model_path}...")
        torch.save(self.model.state_dict(), model_path)
    
    def load_checkpoint(self):
        if not self.checkpoint_path.exists(): 
            logger.warning(f"No checkpoint found at {self.checkpoint_path}. Starting fresh.")
            return
        
        logger.debug(f"Loading checkpoint from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['scheduler'])
        self.current_epoch = ckpt['epoch'] + 1
        self.metrics_tracker.history = ckpt.get('metrics', defaultdict(list))
        self.best_metric_value = ckpt.get('best_metric_value')
        
        if self.scaler and 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])
        
        logger.info(f"Resumed from epoch {self.current_epoch}. Best Metric: {self.best_metric_value}")

    def save_metrics(self):
        self.metrics_tracker.save_to_csv(self.log_path)


class SRTrainer(BaseTrainer):
    
    """Trainer for super-resolution tasks."""
    
    def _init_task_specific(self):
        self.loss_name = 'l1_loss'
        self.lpips_metric = MultispectralLPIPS(self.config)
        logger.debug("SRTrainer initialized. Loss: l1_loss.")
    
    def _get_metric_functions(self) -> Dict[str, Callable]:
        return {
            'psnr': partial(TMF.image.peak_signal_noise_ratio, data_range=(-1, 1), reduction='none', dim=(1, 2, 3)),
            'ssim': partial(TMF.image.structural_similarity_index_measure, data_range=(-1, 1), reduction='none'),
            'mssim': partial(TMF.image.multiscale_structural_similarity_index_measure, data_range=(-1, 1), reduction='none'),
            'lpips': self.lpips_metric,
        }


class DDPMSRTrainer(SRTrainer):
    
    def _init_task_specific(self):
        super()._init_task_specific()
        self.scheduler = DDPMScheduler()
    
    def _compute_loss_and_predictions(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_img, target_img = map(lambda x: x.to(self.device), batch_data)
        logger.debug(f"Input shape: {input_img.shape}, Target shape: {target_img.shape}")
        
        noise = torch.randn_like(target_img)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (input_img.size(0),), device=self.device).long()
        noisy_imgs = self.scheduler.add_noise(target_img, noise, t)
        model_input = torch.cat([noisy_imgs, input_img], dim=1)
        
        with autocast(device_type=self.device.type, dtype=self.hp_dtype, enabled=self.use_amp):
            pred_noise = self.model(model_input, t)
            loss = F.l1_loss(pred_noise, noise, reduction='none').mean(dim=(1, 2, 3))
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"Loss computed. Mean: {loss.mean().item():.4f}, Min: {loss.min().item():.4f}, Max: {loss.max().item():.4f}")
        
        # Reconstruct x_0 (clean image) from x_t and predicted noise
        # Formula: x_0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
        alpha_bar = self.scheduler.alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
        pred_image = (noisy_imgs - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        pred_image = pred_image.clamp(-1.0, 1.0)
        return loss, pred_image, target_img


class FlowMatchingSRTrainer(SRTrainer):

    def _compute_loss_and_predictions(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_img, target_img = map(lambda x: x.to(self.device), batch_data)
        logger.debug(f"Input shape: {input_img.shape}, Target shape: {target_img.shape}")

        x_0 = torch.randn_like(target_img)
        t = torch.rand(input_img.size(0), device=self.device)
        t_batch = t.view(-1, 1, 1, 1)
        x_t = (1 - t_batch) * x_0 + t_batch * target_img
        model_input = torch.cat([x_t, input_img], dim=1)
        
        with autocast(device_type=self.device.type, dtype=self.hp_dtype, enabled=self.use_amp):
            pred_vector = self.model(model_input, t * 1000)
            loss = F.l1_loss(pred_vector, target_img - x_0, reduction='none').mean(dim=(1, 2, 3))
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"Loss computed. Mean: {loss.mean().item():.4f}, Min: {loss.min().item():.4f}, Max: {loss.max().item():.4f}")
        
        pred_image = (pred_vector + x_0).clamp(-1.0, 1.0)
        return loss, pred_image, target_img


class LandCoverTrainer(BaseTrainer):
    """Trainer for land cover classification."""
    
    def _init_task_specific(self):
        
        # self.criterion = DiceLoss(mode='multiclass')
        self.loss_name = 'focal_loss'
        self.num_classes = self.config.get('lc_model', {}).get('num_classes', 7)
        self.act_fn = nn.Sigmoid() if self.num_classes == 1 else nn.Softmax(dim=1)
        
        # disable monitoring for now
        # self.monitor_config = {
            # 'metric': self.config.get('hyperparameters', {}).get('monitor_metric', 'val_ce_loss'),
            # 'mode': self.config.get('hyperparameters', {}).get('monitor_mode', 'min')
        # }
        # logger.debug(f"LandCoverTrainer initialized. Classes: {self.num_classes}, Monitor: {self.monitor_config}")
    
    def _get_metric_functions(self) -> Dict[str, Callable]:
        def samplewise_iou(preds, target):
            """
            Computes IoU sample-by-sample to match the shape (N,) of other metrics.
            """
            # Ensure target is (N, H, W) not (N, 1, H, W)
            if target.ndim == 4 and target.shape[1] == 1:
                target = target.squeeze(1)

            results = []
            for p, t in zip(preds, target):
                # CRITICAL FIX: Unsqueeze to create a 'batch of 1'
                # p: (C, H, W) -> (1, C, H, W)
                # t: (H, W)    -> (1, H, W)
                val = TMF.classification.multiclass_jaccard_index(
                    p.unsqueeze(0), 
                    t.unsqueeze(0), 
                    num_classes=self.num_classes, 
                    average='macro'
                )
                results.append(val)
            
            return torch.stack(results)
            
        return {
            'accuracy': partial(TMF.classification.multiclass_accuracy, num_classes=self.num_classes, average='micro', multidim_average='samplewise'),
            'precision': partial(TMF.classification.multiclass_precision, num_classes=self.num_classes, average='macro', multidim_average='samplewise'),
            'recall': partial(TMF.classification.multiclass_recall, num_classes=self.num_classes, average='macro', multidim_average='samplewise'),
            'f1_score': partial(TMF.classification.multiclass_f1_score, num_classes=self.num_classes, average='macro', multidim_average='samplewise'),
            'miou': samplewise_iou,
        }
    
    def _compute_loss_and_predictions(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, y = map(lambda x: x.to(self.device), batch_data)
        
        with autocast(device_type=self.device.type, dtype=self.hp_dtype, enabled=self.use_amp):
            logits = self.model(X)
            y_pred = self.act_fn(logits)
            loss = focal_loss(y_pred, y, gamma=2.0, reduction='none').mean(dim=1)
            # loss = F.cross_entropy(logits, y.long(), reduction='none').mean(dim=1)
            # loss = self.criterion(logits, y.long()).mean(dim=1)
            logger.debug(f"LC Fwd - Logits: {logits.shape}, Loss Mean: {loss.mean().item():.4f}")
        
        return loss, y_pred, y
    

class RealESRGANTrainer(SRTrainer):
    """
    Trainer for Real-ESRGAN.
    Architecture: RRDBNet (G) + UNetDiscriminatorSN (D).
    Losses: L1 + Perceptual + GAN (BCE/Logits).
    """
    def _init_task_specific(self):
        # 1. Initialize Discriminator (U-Net with Spectral Norm)
        self.discriminator = UNetDiscriminatorSN(self.config).to(self.device)
        logger.debug("Initialized UNetDiscriminatorSN.")

        # 2. Losses
        self.cri_pix = nn.L1Loss().to(self.device)
        self.cri_perceptual = MultispectralPerceptualLoss(self.config).to(self.device)
        self.cri_gan = nn.BCEWithLogitsLoss().to(self.device)
        self.lpips_metric = MultispectralLPIPS(self.config)
        
        # Real-ESRGAN Paper Weights
        self.loss_weights = self.config.get('loss_weights', {
            'pix': 1.0, 
            'percep': 1.0, 
            'gan': 0.1
        })

        # 3. Optimizers
        hp = self.config.get('hyperparameters', {})
        lr_g = hp.get('learning_rate_generator', 1e-4)
        lr_d = hp.get('learning_rate_discriminator', 1e-4)
        
        # Generator Optimizer (alias self.optimizer from BaseTrainer)
        self.optimizer_G = self.optimizer 
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_g
            # CRITICAL FIX: Remove the 'initial_lr' key left by the BaseTrainer's scheduler
            # This forces the new scheduler to capture the new 'lr' as the baseline.
            if 'initial_lr' in param_group:
                del param_group['initial_lr']
            
        # Discriminator Optimizer
        self.optimizer_D = torch.optim.AdamW(
            self.discriminator.parameters(), lr=lr_d, weight_decay=hp.get('weight_decay', 0)
        )
        
        # 4. Schedulers (Warmup + Cosine for BOTH)
        self.scheduler_G = self._create_scheduler_for_optimizer(self.optimizer_G)
        self.scheduler_D = self._create_scheduler_for_optimizer(self.optimizer_D)        
        
        self.loss_name = 'generator_loss'
        
        # Initialize gradients to zero to start accumulation correctly
        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

    def _create_scheduler_for_optimizer(self, optimizer):
        """Helper to create Warmup + Cosine scheduler for any optimizer."""
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs - self.warmup_epochs
        )
        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs - 1
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
            )
        return cosine

    def fit(self, train_dataloader, val_dataloader=None):
        logger.info(f"Starting Real-ESRGAN training.")
        for epoch in range(self.current_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            self.metrics_tracker.reset_epoch()
            
            # Get current LRs for logging
            lr_g = self.optimizer_G.param_groups[0]['lr']
            # lr_d = self.optimizer_D.param_groups[0]['lr']
            self.metrics_tracker.update_epoch(epoch, lr_g) # Log G LR primarily (discriminator LR should be the same)
            
            self._run_phase('train', train_dataloader)
            if val_dataloader:
                self._run_phase('val', val_dataloader)
                
            self.metrics_tracker.finalize_epoch()
            self.scheduler_G.step()
            self.scheduler_D.step()
            self.save_checkpoint()
            self.save_metrics()
            self.save_model()

    def _run_phase(self, phase: str, dataloader: DataLoader):
        is_train = phase == 'train'
        if is_train:
            self.model.train()
            self.discriminator.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            self.discriminator.eval()
            torch.set_grad_enabled(False)
        
        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)
        
        with tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {self.current_epoch}", unit="bt") as pbar:
            for i, batch in enumerate(pbar):
                x, y = map(lambda x: x.to(self.device), batch)
                
                # downsample x back down to original size (if needed, e.g., 256->64)
                # Ensure this matches your data pipeline expectations
                x = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
                
                # discriminator training
                for p in self.discriminator.parameters(): p.requires_grad = True
                
                with autocast(device_type=self.device.type, dtype=self.hp_dtype, enabled=self.use_amp):
                    y_hat = self.model(x)
                    
                    # D Loss: Real
                    d_real = self.discriminator(y)
                    loss_real = self.cri_gan(d_real, torch.ones_like(d_real))
                    
                    # D Loss: Fake
                    d_fake = self.discriminator(y_hat.detach())
                    loss_fake = self.cri_gan(d_fake, torch.zeros_like(d_fake))
                    
                    loss_d = (loss_real + loss_fake) * 0.5

                if is_train:
                    self._backward_step_gan(loss_d, self.optimizer_D, i)

                # generator training
                for p in self.discriminator.parameters(): p.requires_grad = False
                
                with autocast(device_type=self.device.type, dtype=self.hp_dtype, enabled=self.use_amp):
                    # Re-run Discriminator on fake (to recover graph for G)
                    pred_g_fake = self.discriminator(y_hat)
                    
                    # GAN Loss
                    l_g_gan = self.loss_weights['gan'] * self.cri_gan(pred_g_fake, torch.ones_like(pred_g_fake))
                    
                    # Pixel Loss
                    l_g_pix = self.loss_weights['pix'] * self.cri_pix(y_hat, y)
                    
                    # Perceptual Loss (Multispectral)
                    l_g_percep = self.loss_weights['percep'] * self.cri_perceptual(y_hat, y)
                    
                    loss_g = l_g_pix + l_g_percep + l_g_gan

                if is_train:
                    self._backward_step_gan(loss_g, self.optimizer_G, i)

                # Metrics
                batch_metrics = self.metrics_tracker.update_batch(
                    phase, loss_g.detach(), y_hat.detach(), y,
                    additional_metrics={
                        'discriminator_loss': loss_d.detach().item(),
                        'pixel_loss': l_g_pix.detach().item(),
                        'perceptual_loss': l_g_percep.detach().item(),
                        'gan_loss': l_g_gan.detach().item(),
                    }
                )
                
                pbar.set_postfix({
                    'lr': f'{self.optimizer_G.param_groups[0]["lr"]:.2e}',
                    **{k: f'{v:.2e}' for k, v in batch_metrics.items() if k in ['psnr', 'lpips', 'gan_loss', 'discriminator_loss', 'generator_loss']} # only report key metrics
                })

    def _backward_step_gan(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, batch_idx: int):
        """
        Handles Gradient Accumulation and Scaler Logic for a specific optimizer.
        """
        # 1. Scale loss by accumulation steps
        scaled_loss = loss / self.grad_accum_steps
        
        # 2. Backward (Accumulate gradients)
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
            
        # 3. Step (only at accumulation boundaries)
        if (batch_idx + 1) % self.grad_accum_steps == 0:
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            # 4. Zero Gradients (only after stepping)
            optimizer.zero_grad(set_to_none=True)

    def _compute_loss_and_predictions(self, batch_data):
        raise NotImplementedError("Use _run_phase for GAN training.")
    
    def save_checkpoint(self):
        state = {
            'model': self.model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'scheduler_G': self.scheduler_G.state_dict(), # Save G scheduler
            'scheduler_D': self.scheduler_D.state_dict(), # Save D scheduler
            'epoch': self.current_epoch,
            'metrics': self.metrics_tracker.to_dict()
        }
        if self.scaler: state['scaler'] = self.scaler.state_dict()
        torch.save(state, self.checkpoint_path)

    def load_checkpoint(self):
        if not self.checkpoint_path.exists(): return
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if 'discriminator' in ckpt: self.discriminator.load_state_dict(ckpt['discriminator'])
        
        self.optimizer_G.load_state_dict(ckpt['optimizer_G'])
        if 'optimizer_D' in ckpt: self.optimizer_D.load_state_dict(ckpt['optimizer_D'])
        
        # Load Schedulers
        if 'scheduler_G' in ckpt: # Handle legacy checkpoints
            self.scheduler_G.load_state_dict(ckpt['scheduler_G'])
        elif 'scheduler' in ckpt: # Fallback to BaseTrainer key
            self.scheduler_G.load_state_dict(ckpt['scheduler'])
            
        if 'scheduler_D' in ckpt: 
            self.scheduler_D.load_state_dict(ckpt['scheduler_D'])
            
        self.current_epoch = ckpt['epoch'] + 1
        if self.scaler and 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])

    
def train_sr_model(config: Dict[str, Any], model: nn.Module):
    
    load_checkpoint = config.get('job', {}).get('load_checkpoint', False)
    if load_checkpoint:
        logger.info("`load_checkpoint` is True; loading trainer from checkpoint...")
        trainer = FlowMatchingSRTrainer.from_checkpoint(config, model)
    else:
        logger.info("`load_checkpoint` is False; initializing new trainer...")
        trainer = FlowMatchingSRTrainer(config, model)
    
    logger.info("Starting super-resolution model training...")
    trainer.fit(config)
    logger.info("Super-resolution model training complete.")
    


def train_lc_model(config: Dict[str, Any]):
    trainer = LandCoverTrainer()
    logger.fit("Starting land cover model training...")
    trainer.train(config)