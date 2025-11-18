from collections.abc import Callable
import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchmetrics.functional as TMF
from tqdm import tqdm
from functools import partial
from abc import ABC, abstractmethod, ABCMeta
from ..loss import focal_loss
from ..metrics import MultispectralLPIPS, MetricsTracker
from ..utils import get_device, get_hp_dtype

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """Abstract base class for model training."""
    
    def __init__(self, config: Dict[str, Any], model: nn.Module):

        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        
        # Setup paths
        self.out_path = config['paths']['out_path']
        self.checkpoint_path = self.out_path / config.get('job', {}).get('checkpoint_filename', 'checkpoint.pt')
        self.log_path = config['paths']['log_path']
        
        # Load hyperparameters
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
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.init_learning_rate,
            weight_decay=self.weight_decay,
        )
        self.current_epoch = 1
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Setup AMP
        self.hp_dtype = torch.float32
        if self.use_amp:
            self.hp_dtype = get_hp_dtype()
            logger.info(f"Using AMP with dtype: {self.hp_dtype}")
        else:
            logger.info("AMP disabled; using full precision (float32).")
        
        self.scaler = GradScaler(enabled=True) if self.hp_dtype == torch.float16 else None
        
        # Initialize task-specific components
        self._init_task_specific()
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            metric_fns=self._get_metric_functions(),
            phases=('train', 'val')
        )
        
        # Store metric_names for compatibility
        self.metric_names = self.metrics_tracker.metric_names
        
        # Maintain original naming for compatibility
        if hasattr(self, '_use_history_dict') and self._use_history_dict:
            self.history_dict = self.metrics_tracker.history
        else:
            self.metrics = self.metrics_tracker.history
        
        logger.info(f"Initialized {self.__class__.__name__} on device {self.device} with model {type(self.model).__name__}")
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler with warmup."""
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs - self.warmup_epochs,
        )
        
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs - 1,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs],
            )
        else:
            return cosine_scheduler
    
    def _init_task_specific(self):
        """Initialize task-specific components. Override in subclasses."""
        pass
    
    @abstractmethod
    def _get_metric_functions(self) -> Dict[str, Callable]:
        """Return dictionary of metric names to metric functions."""
        pass
    
    @abstractmethod
    def _compute_loss_and_predictions(
        self, 
        batch_data: Tuple[torch.Tensor, ...], 
        phase: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss and return predictions and targets.
        
        Returns:
            Tuple of (loss, predictions, targets)
        """
        pass
    
    @classmethod
    def from_checkpoint(cls, config: Dict[str, Any], model: nn.Module):
        """Create trainer instance from checkpoint."""
        trainer = cls(config, model)
        trainer.load_checkpoint()
        return trainer
    
    def fit(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> None:
        """Train the model."""
        for epoch in range(self.current_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            
            # Update epoch metrics
            self.metrics_tracker.update_epoch(epoch, self.optimizer.param_groups[0]['lr'])
            
            logger.debug(f"Starting epoch {epoch}/{self.num_epochs}...")
            epoch_metrics = self._run_epoch(train_dataloader, val_dataloader)
            
            # Update metrics tracker
            self.metrics_tracker.update_metrics(epoch_metrics)
            
            # Step scheduler and save
            self.lr_scheduler.step()
            self.save_checkpoint()
            self.save_model()
            self.save_metrics()
    
    def _run_epoch(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Run one epoch of training and validation."""
        if val_dataloader is not None:
            phases = ('train', 'val')
            logger.debug("Validation dataloader provided; running both training and validation phases.")
        else:
            phases = ('train',)
            logger.debug("No validation dataloader provided; running only training phase.")
        
        # Initialize epoch metrics
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
            
            samples_seen = 0
            self.optimizer.zero_grad(set_to_none=True)
            
            with tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {self.current_epoch}/{self.num_epochs}", unit="batches") as pbar:
                for batch_num, batch_data in enumerate(pbar):
                    step_metrics = self._step(batch_num, batch_data, phase)
                    samples_seen += batch_data[0].size(0)
                    
                    # Accumulate metrics
                    for k, v in step_metrics.items():
                        epoch_metrics[k] += v
                    
                    # Update progress bar
                    metrics_fmt = {'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'}
                    for metric in self.metric_names:
                        avg = epoch_metrics[f'{phase}_{metric}'] / samples_seen
                        metrics_fmt[metric] = f'{avg:.2e}'
                    pbar.set_postfix(metrics_fmt)
                    
                    logger.debug(f'Epoch {self.current_epoch} {phase} step {batch_num} metrics: ' +
                                '\n'.join([f'{k}: {v}' for k, v in metrics_fmt.items()]))
            
            # Average metrics over samples
            for metric in self.metric_names:
                epoch_metrics[f'{phase}_{metric}'] /= samples_seen
        
        return epoch_metrics
    
    def _step(self, batch_num: int, batch_data: Tuple[torch.Tensor], phase: str) -> Dict[str, float]:
        """Execute a single training/validation step."""
        # Compute loss and get predictions/targets
        loss, predictions, targets = self._compute_loss_and_predictions(batch_data, phase)
        
        # Backward pass (only in training)
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
        
        # Compute metrics using the tracker, passing loss directly
        step_metrics = self.metrics_tracker.compute_metrics(
            loss.detach(),
            predictions.detach(), 
            targets,
            phase
        )
        
        return step_metrics
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        logger.debug(f"Saving checkpoint to {self.checkpoint_path} at epoch {self.current_epoch}...")
        
        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'metrics': self.metrics_tracker.to_dict(),
        }
        
        if self.scaler is not None:
            checkpoint_dict['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint_dict, self.checkpoint_path)
        logger.debug(f"Checkpoint saved to {self.checkpoint_path}.")
    
    def load_checkpoint(self):
        """Load training checkpoint."""
        logger.debug(f"Loading checkpoint from {self.checkpoint_path}...")
        
        if not self.checkpoint_path.exists():
            logger.warning(f"Checkpoint file {self.checkpoint_path} does not exist. Starting from scratch.")
            return
        
        checkpoint_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])
        self.current_epoch = checkpoint_dict['current_epoch'] + 1
        
        if 'metrics' in checkpoint_dict:
            self.metrics_tracker.history = checkpoint_dict['metrics']
            # Update the reference attribute
            if hasattr(self, '_use_history_dict') and self._use_history_dict:
                self.history_dict = self.metrics_tracker.history
            else:
                self.metrics = self.metrics_tracker.history
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {self.checkpoint_path}, resuming from epoch {self.current_epoch}.")
    
    def save_metrics(self):
        """Save metrics to CSV."""
        self.metrics_tracker.save_to_csv(self.log_path)
    
    def save_model(self):
        """Save model weights."""
        model_file = self.out_path / 'model.pt'
        logger.debug(f"Saving model to {model_file}...")
        torch.save(self.model.state_dict(), model_file)
        logger.debug(f"Model saved to {model_file}.")


class FlowMatchingSRTrainer(BaseTrainer):
    """Trainer for super-resolution using flow matching."""
    
    _use_history_dict = False  # Flag to use history_dict instead of metrics
    
    def _init_task_specific(self):
        """Initialize LPIPS metric."""
        self.lpips_metric = MultispectralLPIPS(self.config)
    
    def _get_metric_functions(self) -> Dict[str, Callable]:
        """Return SR-specific metric functions - matches original line 91."""
        return {
            'l1_loss': None,  # Loss is computed in _compute_loss_and_predictions and passed to compute_metrics
            'psnr': partial(
                TMF.image.peak_signal_noise_ratio,
                data_range=(-1, 1),
                reduction='none',
                dim=(1, 2, 3)
            ),
            'ssim': partial(
                TMF.image.structural_similarity_index_measure,
                data_range=(-1, 1),
                reduction='none'
            ),
            'mssim': partial(
                TMF.image.multiscale_structural_similarity_index_measure,
                data_range=(-1, 1),
                reduction='none'
            ),
            'lpips': self.lpips_metric,
        }
    
    def _compute_loss_and_predictions(
        self, 
        batch_data: Tuple[torch.Tensor, ...], 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute flow matching loss - matches original lines 188-217."""
        input_img, target_img = batch_data
        input_img, target_img = input_img.to(self.device), target_img.to(self.device)
        
        # Flow matching setup - exactly as original
        x_0 = torch.randn_like(target_img)
        t = torch.rand(input_img.size(0), device=self.device)
        t_batch = t.view(-1, 1, 1, 1)
        
        # linear interpolation between input and noise to form optimal transport map
        x_t = (1 - t_batch) * x_0 + t_batch * target_img
        target_vector = target_img - x_0
        model_input = torch.cat([x_t, input_img], dim=1)
        
        # Forward pass
        with autocast(device_type=self.device.type, dtype=self.hp_dtype, enabled=self.use_amp):
            pred_vector = self.model(model_input, t * 1000)  # scale t to [0, 1000] for time embedding
            loss = F.l1_loss(pred_vector, target_vector, reduction='none').mean(dim=(1, 2, 3))
        
        # Reconstruct prediction - exactly as original line 217
        pred_image = (pred_vector + x_0).clamp(-1.0, 1.0)
        
        return loss, pred_image, target_img


class LandCoverTrainer(BaseTrainer):
    """Trainer for land cover classification."""
    
    _use_history_dict = False  # Flag to use metrics instead of history_dict

    def _init_task_specific(self):
        """Store num_classes from model."""
        self.num_classes = self.model.num_classes
        self.best_val_loss = float('inf')
        self.best_model_path = self.out_path / 'best_model.pt'
    
    def _get_metric_functions(self) -> Dict[str, Callable]:
        """Return land cover-specific metric functions - matches original line 362."""
        return {
            'focal_loss': None,  # Loss is computed in _compute_loss_and_predictions and passed to compute_metrics
            'accuracy': partial(
                TMF.classification.multiclass_accuracy,
                num_classes=self.num_classes,
                multidim_average='samplewise'
            ),
            'precision': partial(
                TMF.classification.multiclass_precision,
                num_classes=self.num_classes,
                average='macro',
                multidim_average='samplewise'
            ),
            'recall': partial(
                TMF.classification.multiclass_recall,
                num_classes=self.num_classes,
                average='macro',
                multidim_average='samplewise'
            ),
            'f1_score': partial(
                TMF.classification.multiclass_f1_score,
                num_classes=self.num_classes,
                average='macro',
                multidim_average='samplewise'
            ),
            'miou': partial(
                TMF.classification.multiclass_jaccard_index,
                num_classes=self.num_classes,
                average='macro',
                multidim_average='samplewise'
            ),
        }
    
    def _compute_loss_and_predictions(
        self, 
        batch_data: Tuple[torch.Tensor, ...], 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute focal loss - matches original lines 461-468."""
        X, y = batch_data
        X, y = X.to(self.device), y.to(self.device)
        
        # Forward pass
        with autocast(device_type=self.device.type, dtype=self.hp_dtype, enabled=self.use_amp):
            y_pred = self.model(X)
            loss = focal_loss(y_pred, y, gamma=2.0, reduction='none').mean(dim=(1, 2, 3))
        
        return loss, y_pred, y

    def fit(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> None:
        """Train the model with best model saving based on validation loss."""
        for epoch in range(self.current_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            
            # Update epoch metrics
            self.metrics_tracker.update_epoch(epoch, self.optimizer.param_groups[0]['lr'])
            
            logger.debug(f"Starting epoch {epoch}/{self.num_epochs}...")
            epoch_metrics = self._run_epoch(train_dataloader, val_dataloader)
            
            # Update metrics tracker
            self.metrics_tracker.update_metrics(epoch_metrics)
            
            # Save best model if validation loss improved
            if val_dataloader is not None:
                val_focal_loss = epoch_metrics.get('val_focal_loss', None)
                if val_focal_loss is not None and val_focal_loss < self.best_val_loss:
                    self.best_val_loss = val_focal_loss
                    logger.info(f"New best validation loss: {val_focal_loss:.6f} at epoch {epoch}. Saving best model...")
                    torch.save(self.model.state_dict(), self.best_model_path)
                    logger.debug(f"Best model saved to {self.best_model_path}.")
            
            # Step scheduler and save
            self.lr_scheduler.step()
            self.save_checkpoint()
            self.save_model()
            self.save_metrics()
    
    def save_checkpoint(self):
        """Save training checkpoint including best validation loss."""
        logger.debug(f"Saving checkpoint to {self.checkpoint_path} at epoch {self.current_epoch}...")
        
        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'metrics': self.metrics_tracker.to_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scaler is not None:
            checkpoint_dict['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint_dict, self.checkpoint_path)
        logger.debug(f"Checkpoint saved to {self.checkpoint_path}.")
    
    def load_checkpoint(self):
        """Load training checkpoint including best validation loss."""
        logger.debug(f"Loading checkpoint from {self.checkpoint_path}...")
        
        if not self.checkpoint_path.exists():
            logger.warning(f"Checkpoint file {self.checkpoint_path} does not exist. Starting from scratch.")
            return
        
        checkpoint_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])
        self.current_epoch = checkpoint_dict['current_epoch'] + 1
        
        if 'metrics' in checkpoint_dict:
            self.metrics_tracker.history = checkpoint_dict['metrics']
            self.metrics = self.metrics_tracker.history
        
        # Restore best validation loss
        if 'best_val_loss' in checkpoint_dict:
            self.best_val_loss = checkpoint_dict['best_val_loss']
            logger.info(f"Restored best validation loss: {self.best_val_loss:.6f}")
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {self.checkpoint_path}, resuming from epoch {self.current_epoch}.")
    
    
    
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