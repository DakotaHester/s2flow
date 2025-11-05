import os
import logging
from typing import Any, Dict
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from torch.utils.data import DataLoader
from importlib.resources import files

logger = logging.getLogger(__name__)

def load_pca():
    
    assets_path = files("s2flow.assets")
    pca_path = os.path.join(assets_path, "pca.joblib")
    logger.debug(f"Loading PCA data from {pca_path}")
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA data not found at {pca_path}. Please run fit_and_save_pca first.")
    pca = load(pca_path)
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