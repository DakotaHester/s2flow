import os
from typing import Any, Dict
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics import functional as TMF
import numpy as np
import rasterio as rio
import geopandas as gpd
from tqdm import trange
import json

from ..utils import get_device
from ..metrics import MultispectralLPIPS
from ..engine.sampling import get_sampler


def sr_model_evaluation(config: Dict[str, Any], model: nn.Module):
    
    samples_par_path = config.get('data', {}).get('samples_par_path', None)
    if samples_par_path is None:
        raise ValueError("samples_par_path must be specified in the config under 'data.samples_par_path'")
    
    samples_gdf = gpd.read_parquet(samples_par_path)
    val_samples_gdf = samples_gdf[samples_gdf['split'] == 'val'].reset_index(drop=True)
    
    batch_size = config.get('hyperparameters', {}).get('micro_batch_size', 32)
    out_path = os.path.join(config.get('job', {}).get('out_dir', './runs'))
    device = get_device()
    
    sampler = get_sampler(config, model)
    lpips_metric = MultispectralLPIPS(config)
    
    metrics = {} # structure: {sample_id: {metric_name: value, ...}, ...}
    for start_idx in trange(0, len(val_samples_gdf), batch_size, desc="Evaluating SR Model"):
        end_idx = min(start_idx + batch_size, len(val_samples_gdf))
        batch_samples = val_samples_gdf.iloc[start_idx:end_idx]
        
        input_tensors = []
        target_tensors = []
        profiles = []
        for _, sample in batch_samples.iterrows():
            input_path = config['data']['input_dir_path'] / f"{sample['sample_id']:06d}.tif"
            target_path = config['data']['target_dir_path'] / f"{sample['sample_id']:06d}.tif"
            
            input_image = rio.open(input_path).read()  # [C, H, W]
            target_image = rio.open(target_path).read()  # [C, H, W]
            target_profile = rio.open(target_path).profile.copy()
            profiles.append(target_profile) # Save profile for later use
            
            input_tensor = torch.from_numpy(input_image).float()
            target_tensor = torch.from_numpy(target_image).float()
            
            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)
        
        input_batch = torch.stack(input_tensors).to(device)
        target_batch = torch.stack(target_tensors).to(device)
        
        output_batch = sampler.sample(input_batch)
        
        l1_loss = F.l1_loss(output_batch, target_batch, reduction='none').mean(dim=(1, 2, 3)) # per-sample L1 loss
        psnr = TMF.image.peaksignaltonoise_ratio(output_batch, target_batch, data_range=(-1, 1), reduction='none', dim=(1, 2, 3)) # per-sample PSNR
        ssim = TMF.image.ssim(output_batch, target_batch, data_range=(-1, 1), reduction='none') # per-sample SSIM
        mssim = TMF.image.multi_scale_ssim(output_batch, target_batch, data_range=(-1, 1), reduction='none') # per-sample MS-SSIM
        lpips = lpips_metric.forward(output_batch, target_batch, reduction='none') # per-sample LPIPS
        
        for i in range(output_batch.size(0)):
            
            sample = batch_samples.iloc[i]
            sample_id = sample['sample_id']
            
            metrics[sample_id] = {
                'L1': l1_loss[i].item(),
                'PSNR': psnr[i].item(),
                'SSIM': ssim[i].item(),
                'MS-SSIM': mssim[i].item(),
                'LPIPS': lpips[i].item()
            }
            # Save output image
            out_image = output_batch[i].cpu().numpy()
            out_profile = profiles[i]
            with rio.open(os.path.join(out_path, f"{sample_id:06d}_sr.tif"), 'w', **out_profile) as dst:
                dst.write(out_image)
    
    with open(os.path.join(out_path, 'sr_evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # raise NotImplementedError("Super-resolution model evaluation is not yet implemented.")


