import os
import logging
from typing import Any, Dict, Optional
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from torch.utils.data import DataLoader
from importlib.resources import files

logger = logging.getLogger(__name__)


class PCAConvLayer(nn.Module):
    """
    1x1 conv that performs StandardScaler -> PCA projection (frozen).
    Provides helpers to scale the PCA channels to the range expected by LPIPS.
    """
    def __init__(self, config):
        super().__init__()
        
        pca = load_pca(config)
        
        mu = pca['standardscaler'].mean_
        sigma = pca['standardscaler'].scale_
        V_T = pca['pca'].components_.T
        
        logger.debug(f"PCA mean (standardized): {mu}")
        logger.debug(f"PCA scale (standardized): {sigma}")
        logger.debug(f"PCA components shape: {V_T.shape}")
        logger.debug(f'PCA components: {V_T}')
        logger.debug(f'PCA explained variance: {pca["pca"].explained_variance_}')

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
        pca_var = pca['pca'].explained_variance_  # shape (3,)
        pca_std = np.sqrt(pca_var).astype(np.float32)
        # register as buffer so it moves with the module and survives state_dict
        self.register_buffer('pca_std', torch.from_numpy(pca_std).view(1, 3, 1, 1))


    def forward(self, x):
        """Return raw PCA scores (B,3,H,W)."""
        return self.pca_conv(x)

    def to_lpips(self, x, k: Optional[float]=1.0, clamp: bool=False):
        """
        Convert original 4-ch input x -> scaled PCA image suitable for LPIPS.
        - k: number of stds that should map to 1.0 (default=2.0).
        Returns tensor in approximately [-1, 1] (clamped if clamp=True).
        """
        out = self.pca_conv(x)  # B,3,H,W
        std = self.pca_std.to(out.device).type_as(out)
        scaled = out / (k * std)
        return scaled.clamp(-1.0, 1.0) if clamp else scaled


def load_pca(config: Dict[str, Any]) -> Dict[str, Any]:
    
    assets_path = files("s2flow.assets")
    pca_path = os.path.join(assets_path, "pca.joblib")
    logger.debug(f"Loading PCA data from {pca_path}")
    
    if os.path.exists(pca_path):
        pca = load(pca_path)
        
    else:
        logger.info("PCA pipeline not found. Fitting new PCA on training data...")
        from .datasets import get_dataloaders
        train_loader, _ = get_dataloaders(config)
        pca = fit_and_save_pca(train_loader, n_components=3)
    
    logger.info("PCA data loaded successfully.")
    # logger.debug(f'PCA parameters: {pca['pca']}')
    return pca
    

def fit_and_save_pca(dataloader: DataLoader, n_components: int = 3) -> Dict[str, Any]:
    """
    Fits StandardScaler and IncrementalPCA on the provided dataloader and saves to disk.
    """
    assets_path = files("s2flow.assets")
    save_path = os.path.join(assets_path, "pca.joblib")
    if os.path.exists(save_path):
        logger.warning(f"PCA model already exists at {save_path}. It will be overwritten.")

    logger.info(f"Fitting new PCA model. This may take a while...")
    scaler = StandardScaler()
    pca = IncrementalPCA(n_components=n_components)
    
    # NOTE: On the initial run, Mean/Std/PCA are fit using a batch size of 1024 images

    # Pass 1: StandardScaler
    logger.info("Pass 1/2: Fitting StandardScaler")
    for _, naip_batch in tqdm(dataloader, desc="Calculating Mean/Std", 
                              unit="batches", total=len(dataloader)):
        # [B, C, H, W] -> [N, C]
        B, C, H, W = naip_batch.shape
        pixels = naip_batch.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        scaler.partial_fit(pixels)

    # Pass 2: PCA
    logger.info("Pass 2/2: Fitting PCA")
    for _, naip_batch in tqdm(dataloader, desc="Fitting PCA", 
                              unit="batches", total=len(dataloader)):
        B, C, H, W = naip_batch.shape
        pixels = naip_batch.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        pixels_std = scaler.transform(pixels)
        pca.partial_fit(pixels_std)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump({'standardscaler': scaler, 'pca': pca}, save_path)
    logger.info(f"PCA data saved to {save_path}")
    
    return {'standardscaler': scaler, 'pca': pca}