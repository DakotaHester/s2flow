from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional
import pandas as pd
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image.dists import DISTSNetwork
import logging
import torch
from .data.pca import PCAConvLayer
from .utils import get_device, get_hp_dtype
from torch.amp import autocast
from contextlib import nullcontext
from abc import ABC, abstractmethod

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
    """
    Handles metric computation, accumulation, averaging, and history tracking.
    """
    def __init__(
        self, 
        metric_fns: Dict[str, Callable], 
        loss_name: str = 'loss',
        phases: Iterable[str] = ('train', 'val')
    ):
        self.metric_fns = metric_fns
        self.loss_name = loss_name
        self.phases = phases
        self.history: Dict[str, list] = defaultdict(list)
        self.reset_epoch()
        logger.debug(f"MetricsTracker initialized. Loss name: '{self.loss_name}'. Tracking: {list(self.metric_fns.keys())}")

    @property
    def metric_names(self):
        return [self.loss_name] + list(self.metric_fns.keys())

    def reset_epoch(self):
        logger.debug("Resetting metric accumulators for new epoch.")
        self.running_totals = defaultdict(float)
        self.running_counts = defaultdict(int)

    def update_epoch(self, epoch: int, lr: float):
        logger.debug(f"Recording metadata for epoch {epoch}: LR={lr:.2e}")
        self.history['epoch'].append(epoch)
        self.history['lr'].append(lr)

    @torch.no_grad()
    def update_batch(
        self, 
        phase: str, 
        loss: Optional[torch.Tensor], 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Computes metrics for a batch. 
        """
        batch_size = preds.size(0)
        batch_metrics = {}
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[{phase}] Update batch size: {batch_size}")
            logger.debug(f"[{phase}] Preds stats: shape={preds.shape}, min={preds.min():.2f}, max={preds.max():.2f}")
            logger.debug(f"[{phase}] Targets stats: shape={targets.shape}, min={targets.min():.2f}, max={targets.max():.2f}")

        if loss is not None:
            loss_val = loss.float()
            total_loss = loss_val.sum().item() if loss_val.ndim > 0 else loss_val.item() * batch_size
            
            key = f'{phase}_{self.loss_name}'
            self.running_totals[key] += total_loss
            self.running_counts[key] += batch_size
            batch_metrics[self.loss_name] = total_loss / batch_size
            logger.debug(f"[{phase}] Accumulated {self.loss_name}: {batch_metrics[self.loss_name]:.4f}")

        for name, fn in self.metric_fns.items():
            try:
                val = fn(preds, targets)
                
                # Determine if result is scalar (batch mean) or vector (samplewise)
                if val.ndim == 0:
                    total_val = val.item() * batch_size
                else:
                    total_val = val.sum().item()

                key = f'{phase}_{name}'
                self.running_totals[key] += total_val
                self.running_counts[key] += batch_size
                batch_metrics[name] = total_val / batch_size
                logger.debug(f"[{phase}] Metric '{name}' computed: {batch_metrics[name]:.4f}")
                
            except Exception as e:
                logger.error(f"[{phase}] Error computing metric '{name}': {str(e)}", exc_info=True)
                batch_metrics[name] = 0.0

        return {k: self.running_totals[f'{phase}_{k}'] / self.running_counts[f'{phase}_{k}'] for k in batch_metrics.keys()}

    def finalize_epoch(self) -> Dict[str, float]:
        logger.debug("Finalizing epoch metrics...")
        epoch_results = {}
        for key, total in self.running_totals.items():
            count = self.running_counts[key]
            avg = total / count if count > 0 else 0.0
            epoch_results[key] = avg
            self.history[key].append(avg)
            logger.debug(f"Epoch metric '{key}': {avg:.4f} (count={count})")
        return epoch_results

    def save_to_csv(self, save_path: Path):
        file_path = save_path / 'metrics.csv'
        logger.debug(f"Saving metrics history to {file_path}")
        df = pd.DataFrame(self.history)
        df.to_csv(file_path, index=False)

    def to_dict(self):
        return dict(self.history)