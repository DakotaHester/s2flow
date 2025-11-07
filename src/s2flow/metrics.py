from typing import Any, Dict, Optional
from torchmetrics import functional as F
import logging
import torch
from .data.pca import PCAConvLayer
from .utils import get_device

logger = logging.getLogger(__name__)


class MultispectralLPIPS:
    '''
    Apply PCA to multispectral images and then compute LPIPS on the reduced components.
    '''
    def __init__(self, config: Dict[str, Any]) -> None:
            
        self.pca_layer = PCAConvLayer(config).to(get_device())
        self.clamp = config.get('metrics', {}).get('pca_lpips_clamp', False)
        self.k = config.get('metrics', {}).get('pca_lpips_k', 1.0) # 
    
    def forward(self, pred_img: torch.Tensor, naip_img: torch.Tensor, 
                reduction: str='none', **kwargs) -> torch.Tensor:
        """
        pred_img, naip_img: (B,4,H,W) multispectral images
        reduction: 'mean', 'sum', or None
        Returns: LPIPS computed on PCA-reduced images.
        """
        pred_pca = self.pca_layer.to_lpips(pred_img, k=self.k, clamp=self.clamp)  # B,3,H,W
        naip_pca = self.pca_layer.to_lpips(naip_img, k=self.k, clamp=self.clamp)  # B,3,H,W
        
        return F.image.learned_perceptual_image_patch_similarity(
            pred_pca, naip_pca,
            reduction=reduction, **kwargs
        )  # (B,) if reduction='none'