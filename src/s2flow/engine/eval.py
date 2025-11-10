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

from ..utils import get_device
from ..metrics import MultispectralLPIPS
from ..engine.sampling import get_sampler
from ..data.utils import scale

logger = getLogger(__name__)


@torch.no_grad()
def sr_model_evaluation(config: Dict[str, Any], model: nn.Module):
    
    model.eval()
    samples_par_path = config.get('data', {}).get('samples_par_path', None)
    if samples_par_path is None:
        raise ValueError("samples_par_path must be specified in the config under 'data.samples_par_path'")
    
    data_dir_path = Path(config.get('data', {}).get('data_dir_path', './data'))
    
    samples_gdf = gpd.read_parquet(samples_par_path)
    val_samples_gdf = samples_gdf[samples_gdf['split'] == 'val'].reset_index(drop=True)
    logger.info(f"Running inference on {len(val_samples_gdf)} validation samples...")
    
    batch_size = config.get('hyperparameters', {}).get('micro_batch_size', 32)
    out_path = config['paths']['out_path']
    image_out_path = out_path / 'sr_outputs'
    image_out_path.mkdir(parents=True, exist_ok=True)
    
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
        filenames = []
        for _, sample in batch_samples.iterrows():
            input_path = data_dir_path / sample['input_path']
            target_path = data_dir_path / sample['target_path']
            
            input_image = rio.open(input_path).read()  # [C, H, W]
            target_image = rio.open(target_path).read()  # [C, H, W]
            target_profile = rio.open(target_path).profile.copy()
            profiles.append(target_profile) # Save profile for later use
            
            input_tensor = scale(torch.from_numpy(input_image).float(), in_range=(0, 10000), out_range=(-1.0, 1.0))
            target_tensor = scale(torch.from_numpy(target_image).float(), in_range=(0, 10000), out_range=(-1.0, 1.0))
            
            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)
            filenames.append(Path(input_path).name)
        
        input_batch = torch.stack(input_tensors).to(device)
        target_batch = torch.stack(target_tensors).to(device)
        
        output_batch = sampler.sample(input_batch)
        
        l1_loss = F.l1_loss(output_batch, target_batch, reduction='none').mean(dim=(1, 2, 3)) # per-sample L1 loss
        psnr = TMF.image.peak_signal_noise_ratio(output_batch, target_batch, data_range=(-1, 1), reduction='none', dim=(1, 2, 3)) # per-sample PSNR
        ssim = TMF.image.structural_similarity_index_measure(output_batch, target_batch, data_range=(-1, 1), reduction='none') # per-sample SSIM
        mssim = TMF.image.multiscale_structural_similarity_index_measure(output_batch, target_batch, data_range=(-1, 1), reduction='none') # per-sample MS-SSIM
        lpips = lpips_metric(output_batch, target_batch) # per-sample LPIPS
        
        output_batch = scale(output_batch.cpu(), in_range=(-1.0, 1.0), out_range=(0, 10000)).numpy()
        for i in range(output_batch.shape[0]):
            
            sample = batch_samples.iloc[i]
            sample_id = sample['id']
            
            metrics[sample_id] = {
                'L1': l1_loss[i].item(),
                'PSNR': psnr[i].item(),
                'SSIM': ssim[i].item(),
                'MS-SSIM': mssim[i].item(),
                'LPIPS': lpips[i].item()
            }
            # Save output image
            out_image = output_batch[i]
            out_profile = profiles[i].copy()
            with rio.open(image_out_path / filenames[i], 'w', **out_profile) as dst:
                dst.write(out_image)
    
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.index.name = 'sample_id'
    metrics_df.to_csv(out_path / 'sr_evaluation_metrics.csv')
    logger.info(f"Saved image-wise SR evaluation metrics to {out_path / 'sr_evaluation_metrics.csv'}")
    
    # caluclate mean, median, std, variance, etc. for each metric
    summary_stats = metrics_df.describe().transpose()
    summary_stats.index.name = 'metric'
    summary_stats.to_csv(out_path / 'sr_evaluation_summary_stats.csv')
    logger.info(f"Saved summary SR evaluation statistics to {out_path / 'sr_evaluation_summary_stats.csv'}")
    
    logger.info('Mean SR Evaluation Metrics:' + f"\n{summary_stats['mean']}")
    # raise NotImplementedError("Super-resolution model evaluation is not yet implemented.")


