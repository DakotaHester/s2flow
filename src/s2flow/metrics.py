from typing import Any, Dict, Optional
from torchmetrics import functional as F
import logging
import torch
import torch.nn as nn
import numpy as np
from .data import pca

logger = logging.getLogger(__name__)

class PCAConvLayer(nn.Module):
    """
    1x1 conv that performs StandardScaler -> PCA projection (frozen).
    Provides helpers to scale the PCA channels to the range expected by LPIPS.
    """
    def __init__(self, pca_pipeline):
        super().__init__()
        
        mu = pca_pipeline['standardscaler'].mean_
        sigma = pca_pipeline['standardscaler'].scale_
        V_T = pca_pipeline['pca'].components_.T
        
        logger.debug(f"PCA mean (standardized): {mu}")
        logger.debug(f"PCA scale (standardized): {sigma}")
        logger.debug(f"PCA components shape: {V_T.shape}")
        logger.debug(f'PCA components: {V_T}')
        logger.debug(f'PCA explained variance: {pca_pipeline["pca"].explained_variance_}')

        # weight/bias that map original 4-channel image -> PCA scores
        W = np.diag(1.0 / sigma) @ V_T
        b = - (mu / sigma) @ V_T

        self.pca_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, bias=True)

        weight_tensor = torch.from_numpy(W.T).float().unsqueeze(-1).unsqueeze(-1)
        bias_tensor = torch.from_numpy(b).float()

        with torch.no_grad():
            self.pca_conv.weight.copy_(weight_tensor)
            self.pca_conv.bias.copy_(bias_tensor)

        # freeze parameters
        for p in self.pca_conv.parameters():
            p.requires_grad = False

        # store per-component std (explained variance of PCA on standardized data)
        pca_var = pca_pipeline['pca'].explained_variance_  # shape (3,)
        pca_std = np.sqrt(pca_var).astype(np.float32)
        # register as buffer so it moves with the module and survives state_dict
        self.register_buffer('pca_std', torch.from_numpy(pca_std).view(1, 3, 1, 1))


    def forward(self, x):
        """Return raw PCA scores (B,3,H,W)."""
        return self.pca_conv(x)

    def to_lpips(self, x, k: Optional[float]=2.0, clamp: bool=True):
        """
        Convert original 4-ch input x -> scaled PCA image suitable for LPIPS.
        - k: number of stds that should map to 1.0 (default=2.0).
        Returns tensor in approximately [-1, 1] (clamped if clamp=True).
        """
        out = self.pca_conv(x)  # B,3,H,W
        std = self.pca_std.to(out.device).type_as(out)
        scaled = out / (k * std)
        return scaled.clamp(-1.0, 1.0) if clamp else scaled


class MultispectralLPIPS:
    '''
    Apply PCA to multispectral images and then compute LPIPS on the reduced components.
    '''
    def __init__(self, config: Dict[str, Any]) -> None:
        
        try:
            self.pca = pca.load_pca()
        except FileNotFoundError:
            logger.info("PCA pipeline not found. Fitting new PCA on training data...")
            from .data.datasets import get_dataloaders
            train_loader, _ = get_dataloaders(config)
            self.pca = pca.fit_and_save_pca(train_loader, n_components=3)
            
        self.pca_layer = PCAConvLayer(self.pca)
        self.k = config.get('metrics', {}).get('pca_lpips_k', 2.0) # 
    
    def forward(self, pred_img: torch.Tensor, naip_img: torch.Tensor, 
                reduction: str='none', **kwargs) -> torch.Tensor:
        """
        pred_img, naip_img: (B,4,H,W) multispectral images
        reduction: 'mean', 'sum', or None
        Returns: LPIPS computed on PCA-reduced images.
        """
        pred_pca = self.pca_layer.to_lpips(pred_img, k=self.k)  # B,3,H,W
        naip_pca = self.pca_layer.to_lpips(naip_img, k=self.k)  # B,3,H,W
        
        return F.image.learned_perceptual_image_patch_similarity(
            pred_pca, naip_pca,
            reduction=reduction, **kwargs
        )  # (B,) if reduction='none'