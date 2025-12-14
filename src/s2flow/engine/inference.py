from contextlib import nullcontext
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


@torch.no_grad()
def simple_lc_model_inference(config: Dict[str, Any], model: nn.Module) -> None:
    """Simple inference routine for land cover classification.
    
    This function performs batch inference on a directory of input images
    and saves the predicted land cover maps.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        model (nn.Module): Land cover classification model.
    """
    
    model.eval()
    device = get_device()
    
    data_root_path = Path(config.get('data', {}).get('data_root_path', './data'))
    glob_pattern = config.get('data', {}).get('glob_pattern', '*.tif')
    out_root_path = Path(config.get('data', {}).get('out_root_path', './out'))
    out_root_path.mkdir(parents=True, exist_ok=True)
    
    input_files = sorted(data_root_path.glob(glob_pattern))
    logger.info(f"Found {len(input_files)} input files for LC inference.")
    
    if len(input_files) == 0:
        logger.warning(f"No files found matching pattern '{glob_pattern}' in {data_root_path}")
        return
    
    batch_size = config.get('hyperparameters', {}).get('micro_batch_size', 8)
    num_classes = config.get('lc_model', {}).get('num_classes', 7)
    
    # Determine input data range based on source type
    source_data = config.get('data', {}).get('source_data', 's2')
    if source_data in ('s2', 's2sr'):
        in_range = (0, 10000)
    elif source_data == 'naip':
        in_range = (0, 255)
    else:
        logger.warning(f"Unknown source_data '{source_data}', defaulting to (0, 10000)")
        in_range = (0, 10000)
    
    logger.info(f"Input data type: {source_data}, scaling from {in_range} to (0, 1)")
    
    # Get colormap if specified
    colormap = config.get('inference', {}).get('colormap', None)
    save_colormap = config.get('inference', {}).get('save_colormap', True)
    
    # AMP context
    if config.get('hyperparameters', {}).get('use_amp', True):
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=get_hp_dtype())
        logger.info(f"Using AMP with dtype: {get_hp_dtype()}")
    else:
        autocast_ctx = nullcontext()
        logger.info("AMP disabled; using full precision.")
    
    gpu_time = 0.0
    total_start_time = time()
    
    logger.info(f"Starting LC inference with batch size {batch_size} for {len(input_files)} files...")
    with trange(0, len(input_files), batch_size, desc="LC Inference") as pbar:
        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, len(input_files))
            batch_files = input_files[start_idx:end_idx]
            
            input_tensors = []
            profiles = []
            filepaths = []
            
            for input_path in batch_files:
                with rio.open(input_path) as src:
                    input_image = src.read()  # [C, H, W]
                    profile = src.profile.copy()
                
                input_tensor = scale(
                    torch.from_numpy(input_image).float(),
                    in_range=in_range,
                    out_range=(0.0, 1.0)
                )
                
                input_tensors.append(input_tensor)
                profiles.append(profile)
                filepaths.append(input_path.relative_to(data_root_path))
            
            input_batch = torch.stack(input_tensors).to(device)
            
            gpu_start_time = time()
            with autocast_ctx:
                logits_batch = model(input_batch)
            pred_batch = torch.argmax(logits_batch, dim=1)
            gpu_stop_time = time()
            gpu_time += (gpu_stop_time - gpu_start_time)
            
            pred_batch_cpu = pred_batch.cpu().numpy()
            
            for i in range(pred_batch_cpu.shape[0]):
                out_image = (pred_batch_cpu[i] + 1).astype('uint8')  # 1-indexed classes
                out_profile = profiles[i].copy()
                out_profile.update(
                    dtype='uint8',
                    count=1,
                    photometric='palette' if save_colormap else None
                )
                
                # Remove photometric if None
                if out_profile.get('photometric') is None:
                    out_profile.pop('photometric', None)
                
                out_path = out_root_path / filepaths[i]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                with rio.open(out_path, 'w', **out_profile) as dst:
                    dst.write(out_image, 1)
                    if save_colormap and colormap is not None:
                        dst.write_colormap(1, colormap)
    
    total_end_time = time()
    total_time = total_end_time - total_start_time
    
    logger.info(f"LC inference completed. Outputs saved to {out_root_path}.")
    logger.info(f"Total inference time: {total_time:.2f} seconds")
    logger.info(f"Total GPU time: {gpu_time:.2f} seconds")
    logger.info(f"Average time per image: {total_time / len(input_files):.4f} seconds")
    
    # Save timing information
    times_dict = {
        'total_time_seconds': total_time,
        'gpu_inference_time_seconds': gpu_time,
        'num_images': len(input_files),
        'avg_time_per_image': total_time / len(input_files)
    }
    
    times_path = out_root_path / 'lc_inference_times.json'
    with open(times_path, 'w') as f:
        json.dump(times_dict, f, indent=4)
    logger.info(f"Saved timing information to {times_path}")