from typing import Any, Dict, Optional
from torchmetrics import functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import logging
import torch
from .data.pca import PCAConvLayer
from .utils import get_device, get_hp_dtype
from torch.amp import autocast
from contextlib import nullcontext

logger = logging.getLogger(__name__)


class MultispectralLPIPS:
    '''
    Apply PCA to multispectral images and then compute LPIPS on the reduced components.
    '''
    def __init__(self, config: Dict[str, Any]) -> None:
        
        self.device = get_device()
        self.pca_layer = PCAConvLayer(config).to(self.device)
        self.clamp = config.get('metrics', {}).get('pca_lpips_clamp', False)
        self.k = config.get('metrics', {}).get('pca_lpips_k', 1.0)
        self.tv_metric = LearnedPerceptualImagePatchSimilarity(reduction='none').to(self.device)
        
        self.use_amp = config.get('hyperparameters', None).get('use_amp', True)
        if self.use_amp:
            hp_dtype = get_hp_dtype()
            logger.debug(f"Using AMP with dtype: {hp_dtype}")
            self.autocast_context = autocast(device_type=self.device.type, dtype=hp_dtype, enabled=True)
        else:
            logger.debug("AMP disabled; using full precision (float32).")
            self.autocast_context = nullcontext()

    @torch.no_grad()
    def __call__(self, pred_img: torch.Tensor, naip_img: torch.Tensor) -> torch.Tensor:
        """
        pred_img, naip_img: (B,4,H,W) multispectral images
        Returns: LPIPS computed on PCA-reduced images.
        """
        with self.autocast_context:
            pred_pca = self.pca_layer(pred_img, k=self.k, clamp=self.clamp)  # B,3,H,W
            naip_pca = self.pca_layer(naip_img, k=self.k, clamp=self.clamp)  # B,3,H,W
            
            lpips = self.tv_metric(pred_pca, naip_pca)
        self.tv_metric.reset()
        return lpips  # B, tensor