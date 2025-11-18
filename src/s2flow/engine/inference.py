import json
from typing import Any, Dict
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics import functional as TMF
import rasterio as rio
import geopandas as gpd
import pandas as pd
from tqdm import trange
from logging import getLogger
from pathlib import Path
from time import time

from ..utils import get_device
from ..metrics import MultispectralLPIPS, MultispectralDISTS
from ..engine.sampling import get_sampler
from ..data.utils import scale

logger = getLogger(__name__)


@torch.no_grad()
def simple_sr_model_inference(config: Dict[str, Any], model: nn.Module) -> None:
    """ Simple inference routine for super-resolution using the provided model.
    
    NOTE: for simplicity we assume that the input data has already been 
    resampled to the desired resolution. The reason for this is that resampling
    for the land cover training data was done using reproject_match to ensure 
    approximate alignment between S2, NAIP, and the LC labels.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        model (nn.Module): Super-resolution model to use for inference.
    """
    
    model.eval()
    data_root_path = Path(config.get('data', {}).get('data_root_path', './data'))
    glob_pattern = config.get('data', {}).get('glob_pattern', '*.tif')
    out_root_path = Path(config.get('data', {}).get('out_root_path', './out'))
    out_root_path.mkdir(parents=True, exist_ok=True)
    
    input_files = sorted(data_root_path.glob(glob_pattern))
    logger.info(f"Found {len(input_files)} input files for inference.")
    
    device = get_device()
    sampler = get_sampler(config, model)
    batch_size = config.get('hyperparameters', {}).get('micro_batch_size', 8)
    
    logger.info(f"Starting SR inference with batch size {batch_size} for {len(input_files)} files...")
    with trange(0, len(input_files), batch_size, desc="SR Inference") as pbar:
        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, len(input_files))
            batch_files = input_files[start_idx:end_idx]
            
            input_tensors = []
            profiles = []
            filepaths = []
            
            for input_path in batch_files:
                # Use context manager and read file once
                with rio.open(input_path) as src:
                    input_image = src.read()  # [C, H, W]
                    profile = src.profile.copy()
                
                input_tensor = scale(
                    torch.from_numpy(input_image).float(), 
                    in_range=(0, 10000), 
                    out_range=(-1.0, 1.0)
                )
                
                input_tensors.append(input_tensor)
                profiles.append(profile)
                filepaths.append(input_path.relative_to(data_root_path))
            
            input_batch = torch.stack(input_tensors).to(device)
            
            output_batch = sampler.sample(input_batch)
            output_batch = scale(
                output_batch.cpu(), 
                in_range=(-1.0, 1.0), 
                out_range=(0, 10000)
            ).numpy()
            
            for i in range(output_batch.shape[0]):
                out_image = output_batch[i]
                out_profile = profiles[i].copy()
                out_path = out_root_path / filepaths[i]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                with rio.open(out_path, 'w', **out_profile) as dst:
                    dst.write(out_image)
    
    logger.info(f"SR inference completed. Outputs saved to {out_root_path}.")