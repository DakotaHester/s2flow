from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import pandas as pd
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image.dists import DISTSNetwork
import logging
import torch
from .data.pca import PCAConvLayer
from .utils import get_device, get_hp_dtype
from torch.amp import autocast
from contextlib import nullcontext
from abc import ABC, abstractmethod, ABCMeta

logger = logging.getLogger(__name__)


class BaseMultispectralMetric(ABC):
    
    def __init__(self, config: Dict[str, Any]) -> None:
        
        self.device = get_device()
        self.pca_layer = PCAConvLayer(config).to(self.device)
        self.clamp = config.get('metrics', {}).get('pca_lpips_clamp', False)
        self.k = config.get('metrics', {}).get('pca_lpips_k', 1.0)
        
        self.use_amp = config.get('hyperparameters', None).get('use_amp', True)
        if self.use_amp:
            hp_dtype = get_hp_dtype()
            logger.debug(f"Using AMP with dtype: {hp_dtype}")
            self.autocast_context = autocast(device_type=self.device.type, dtype=hp_dtype, enabled=True)
        else:
            logger.debug("AMP disabled; using full precision (float32).")
            self.autocast_context = nullcontext()
    
    @abstractmethod
    @torch.no_grad()
    def __call__(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        pass


class MultispectralLPIPS(BaseMultispectralMetric):
    '''Apply PCA to multispectral images and then compute LPIPS on the reduced components.'''
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.tv_metric = LearnedPerceptualImagePatchSimilarity(reduction='none').to(self.device)
    
    @torch.no_grad()
    def __call__(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """
        pred_img, target_img: (B,4,H,W) multispectral images
        Returns: LPIPS computed on PCA-reduced images.
        """
        with self.autocast_context:
            pred_pca = self.pca_layer(pred_img, k=self.k, clamp=self.clamp)  # B,3,H,W
            naip_pca = self.pca_layer(target_img, k=self.k, clamp=self.clamp)  # B,3,H,W
            
            lpips = self.tv_metric(pred_pca, naip_pca)
        self.tv_metric.reset()
        return lpips  # B, tensor


class SampleWiseDISTSNetwork(DISTSNetwork):
    '''DISTSNetwork that returns per-sample scores for a batch.'''
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)

        dist1 = torch.zeros(B, 1, device=x.device)
        dist2 = torch.zeros(B, 1, device=x.device)

        c1 = 1e-6
        c2 = 1e-6

        # --- normalize weights ---
        w_sum = self.alpha.sum() + self.beta.sum()

        # IMPORTANT: ensure batch-independent broadcasting
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta  = torch.split(self.beta  / w_sum, self.chns, dim=1)

        # ENFORCE SHAPE: (1, C_k, 1, 1)
        alpha = [a for a in alpha]
        beta  = [b for b in beta]

        for k in range(len(self.chns)):
            # Means
            x_mean = feats0[k].mean(dim=[2, 3], keepdim=True)
            y_mean = feats1[k].mean(dim=[2, 3], keepdim=True)

            s1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)

            # (B, C, H, W) * (1, C, 1, 1) → (B, C, H, W) → sum over C → (B,1,1)
            d1 = (alpha[k] * s1).sum(1, keepdim=True)
            dist1 += d1.view(B, 1)

            # Variances and covariance
            x_var  = ((feats0[k] - x_mean)**2).mean(dim=[2, 3], keepdim=True)
            y_var  = ((feats1[k] - y_mean)**2).mean(dim=[2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean(dim=[2, 3], keepdim=True) - x_mean * y_mean

            s2 = (2 * xy_cov + c2) / (x_var + y_var + c2)

            d2 = (beta[k] * s2).sum(1, keepdim=True)
            dist2 += d2.view(B, 1)

        return (1 - (dist1 + dist2)).view(B)


class MultispectralDISTS(BaseMultispectralMetric):
    ''' Apply PCA to multispectral images and then compute DISTS on the reduced components. '''
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)          # BaseMultispectralMetric
        
        self.dists_network = SampleWiseDISTSNetwork().to(self.device)
    
    @torch.no_grad()
    def __call__(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """
        pred_img, target_img: (B,4,H,W) multispectral images
        Returns: DISTS computed on PCA-reduced images.
        """
        with self.autocast_context:
            pred_pca = self.pca_layer(pred_img, k=self.k, clamp=self.clamp)  # B,3,H,W
            naip_pca = self.pca_layer(target_img, k=self.k, clamp=self.clamp)  # B,3,H,W
            
            dists = self.dists_network(pred_pca, naip_pca)
        return dists  # B, tensor


class MetricsTracker:
    """Tracks and manages training/validation metrics across epochs."""
    
    def __init__(
        self, 
        metric_fns: Dict[str, Callable], 
        phases: Tuple[str, ...] = ('train', 'val')
    ):
        """
        Initialize metrics tracker.
        
        Args:
            metric_fns: Dictionary mapping metric names to callable functions
            phases: Tuple of phase names (e.g., 'train', 'val')
        """
        self.metric_fns = metric_fns
        self.metric_names = list(metric_fns.keys())
        self.phases = phases
        self.history = {
            'epoch': [],
            'lr': [],
        }
        
        for phase in phases:
            for metric in self.metric_names:
                self.history[f'{phase}_{metric}'] = []
    
    def compute_metrics(
        self, 
        loss: torch.Tensor,
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        phase: str
    ) -> Dict[str, float]:
        """
        Compute all metrics for a batch.
        
        Args:
            loss: Already computed loss tensor (per-sample)
            predictions: Model predictions
            targets: Ground truth targets
            phase: Current phase ('train' or 'val')
            
        Returns:
            Dictionary mapping '{phase}_{metric_name}' to summed metric values
        """
        metrics = {}
        
        for name, metric_fn in self.metric_fns.items():
            if name.endswith('loss'):
                # Loss is already computed, just use it
                value = loss
            else:
                # Compute other metrics
                value = metric_fn(predictions, targets)
            
            # Sum over batch dimension for accumulation
            metrics[f'{phase}_{name}'] = value.sum().item() if isinstance(value, torch.Tensor) else value
        
        return metrics
    
    def update_epoch(self, epoch: int, lr: float):
        """Update epoch and learning rate."""
        self.history['epoch'].append(epoch)
        self.history['lr'].append(lr)
    
    def update_metrics(self, epoch_metrics: Dict[str, float]):
        """Update metrics from epoch results."""
        for phase in self.phases:
            for metric in self.metric_names:
                key = f'{phase}_{metric}'
                self.history[key].append(epoch_metrics.get(key, None))
    
    def get_running_metrics(
        self, 
        phase: str, 
        samples_seen: int, 
        accumulated_metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Get formatted metrics for progress bar."""
        metrics_fmt = {}
        for metric in self.metric_names:
            key = f'{phase}_{metric}'
            if key in accumulated_metrics:
                avg = accumulated_metrics[key] / samples_seen
                metrics_fmt[metric] = f'{avg:.2e}'
        return metrics_fmt
    
    def save_to_csv(self, log_path: Path, filename: str = 'training_metrics.csv'):
        """Save metrics history to CSV file."""
        csv_path = log_path / filename
        logger.debug(f"Saving training metrics to {csv_path}...")
        pd.DataFrame(self.history).to_csv(csv_path, index=False)
    
    def to_dict(self) -> Dict[str, List]:
        """Return metrics history as dictionary."""
        return self.history.copy()