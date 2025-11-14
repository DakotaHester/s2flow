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
from ..metrics import MultispectralLPIPS
from ..engine.sampling import get_sampler
from ..data.utils import scale

logger = getLogger(__name__)


@torch.no_grad()
def sr_model_evaluation(config: Dict[str, Any], model: nn.Module):
    
    logger.debug("Starting SR model evaluation...")
    model.eval()
    logger.debug(f"Model set to evaluation mode: {not model.training}")
    
    samples_par_path = config.get('data', {}).get('samples_par_path', None)
    if samples_par_path is None:
        raise ValueError("samples_par_path must be specified in the config under 'data.samples_par_path'")
    logger.debug(f"Samples parquet path: {samples_par_path}")
    
    data_dir_path = Path(config.get('data', {}).get('data_dir_path', './data'))
    logger.debug(f"Data directory path: {data_dir_path}")
    
    logger.debug(f"Loading samples from {samples_par_path}...")
    samples_gdf = gpd.read_parquet(samples_par_path)
    logger.debug(f"Loaded {len(samples_gdf)} total samples")
    logger.debug(f"Split distribution: {samples_gdf['split'].value_counts().to_dict()}")
    
    val_samples_gdf = samples_gdf[samples_gdf['split'] == 'val'].reset_index(drop=True)
    logger.info(f"Running inference on {len(val_samples_gdf)} validation samples...")
    logger.debug(f"Validation samples index range: {val_samples_gdf.index.min()} to {val_samples_gdf.index.max()}")
    
    batch_size = config.get('hyperparameters', {}).get('micro_batch_size', 32)
    logger.debug(f"Using batch size: {batch_size}")
    
    out_path = config['paths']['out_path']
    logger.debug(f"Output path: {out_path}")
    
    image_out_path = out_path / 'sr_outputs'
    image_out_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created image output directory: {image_out_path}")
    
    device = get_device()
    logger.debug(f"Using device: {device}")
    
    logger.debug("Initializing sampler...")
    sampler = get_sampler(config, model)
    logger.debug(f"Sampler initialized: {type(sampler).__name__}")
    
    logger.debug("Initializing LPIPS metric...")
    lpips_metric = MultispectralLPIPS(config)
    logger.debug("LPIPS metric initialized")
    
    gpu_time = 0.0
    total_start_time = time()
    logger.debug(f"Evaluation start time: {total_start_time}")
    
    try:
        metrics = {} # structure: {sample_id: {metric_name: value, ...}, ...}
        num_batches = (len(val_samples_gdf) + batch_size - 1) // batch_size
        logger.debug(f"Total batches to process: {num_batches}")
        
        with trange(0, len(val_samples_gdf), batch_size, desc="Evaluating SR Model") as pbar:
            for batch_idx, start_idx in enumerate(pbar):
                end_idx = min(start_idx + batch_size, len(val_samples_gdf))
                actual_batch_size = end_idx - start_idx
                logger.debug(f"\nBatch {batch_idx + 1}/{num_batches}: Processing samples {start_idx} to {end_idx - 1} (size: {actual_batch_size})")
                
                batch_samples = val_samples_gdf.iloc[start_idx:end_idx]
                
                input_tensors = []
                target_tensors = []
                profiles = []
                filenames = []
                
                logger.debug(f"Loading {actual_batch_size} image pairs...")
                for sample_idx, (_, sample) in enumerate(batch_samples.iterrows()):
                    input_path = data_dir_path / sample['input_path']
                    target_path = data_dir_path / sample['target_path']
                    
                    logger.debug(f"  Sample {sample_idx + 1}/{actual_batch_size} (ID: {sample['id']})")
                    logger.debug(f"    Input: {input_path}")
                    logger.debug(f"    Target: {target_path}")
                    
                    input_image = rio.open(input_path).read()  # [C, H, W]
                    target_image = rio.open(target_path).read()  # [C, H, W]
                    target_profile = rio.open(target_path).profile.copy()
                    
                    logger.debug(f"    Input shape: {input_image.shape}, dtype: {input_image.dtype}")
                    logger.debug(f"    Target shape: {target_image.shape}, dtype: {target_image.dtype}")
                    logger.debug(f"    Input range: [{input_image.min()}, {input_image.max()}]")
                    logger.debug(f"    Target range: [{target_image.min()}, {target_image.max()}]")
                    
                    profiles.append(target_profile) # Save profile for later use
                    
                    input_tensor = scale(torch.from_numpy(input_image).float(), in_range=(0, 10000), out_range=(-1.0, 1.0))
                    target_tensor = scale(torch.from_numpy(target_image).float(), in_range=(0, 10000), out_range=(-1.0, 1.0))
                    
                    logger.debug(f"    Scaled input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
                    logger.debug(f"    Scaled target range: [{target_tensor.min():.4f}, {target_tensor.max():.4f}]")
                    
                    input_tensors.append(input_tensor)
                    target_tensors.append(target_tensor)
                    filenames.append(Path(input_path).name)
                
                logger.debug(f"Stacking {len(input_tensors)} tensors into batch...")
                input_batch = torch.stack(input_tensors).to(device)
                target_batch = torch.stack(target_tensors).to(device)
                logger.debug(f"Input batch shape: {input_batch.shape}, device: {input_batch.device}")
                logger.debug(f"Target batch shape: {target_batch.shape}, device: {target_batch.device}")
                
                logger.debug("Starting GPU sampling...")
                gpu_start_time = time()
                output_batch = sampler.sample(input_batch)
                gpu_stop_time = time()
                batch_gpu_time = gpu_stop_time - gpu_start_time
                gpu_time += batch_gpu_time
                logger.debug(f"GPU sampling completed in {batch_gpu_time:.4f} seconds")
                logger.debug(f"Output batch shape: {output_batch.shape}, device: {output_batch.device}")
                logger.debug(f"Output range: [{output_batch.min():.4f}, {output_batch.max():.4f}]")
                
                logger.debug("Computing metrics...")
                l1_loss = F.l1_loss(output_batch, target_batch, reduction='none').mean(dim=(1, 2, 3)) # per-sample L1 loss
                logger.debug(f"L1 loss computed: mean={l1_loss.mean():.6f}, std={l1_loss.std():.6f}")
                
                psnr = TMF.image.peak_signal_noise_ratio(output_batch, target_batch, data_range=(-1, 1), reduction='none', dim=(1, 2, 3)) # per-sample PSNR
                logger.debug(f"PSNR computed: mean={psnr.mean():.4f}, std={psnr.std():.4f}")
                
                ssim = TMF.image.structural_similarity_index_measure(output_batch, target_batch, data_range=(-1, 1), reduction='none') # per-sample SSIM
                logger.debug(f"SSIM computed: mean={ssim.mean():.4f}, std={ssim.std():.4f}")
                
                mssim = TMF.image.multiscale_structural_similarity_index_measure(output_batch, target_batch, data_range=(-1, 1), reduction='none') # per-sample MS-SSIM
                logger.debug(f"MS-SSIM computed: mean={mssim.mean():.4f}, std={mssim.std():.4f}")
                
                lpips = lpips_metric(output_batch, target_batch) # per-sample LPIPS
                logger.debug(f"LPIPS computed: mean={lpips.mean():.4f}, std={lpips.std():.4f}")
                
                logger.debug("Scaling output batch back to original range and saving images...")
                output_batch = scale(output_batch.cpu(), in_range=(-1.0, 1.0), out_range=(0, 10000)).numpy()
                logger.debug(f"Scaled output range: [{output_batch.min():.2f}, {output_batch.max():.2f}]")
                
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
                    logger.debug(f"  Sample {sample_id} metrics: L1={l1_loss[i].item():.6f}, PSNR={psnr[i].item():.4f}, SSIM={ssim[i].item():.4f}, MS-SSIM={mssim[i].item():.4f}, LPIPS={lpips[i].item():.4f}")
                    
                    # Save output image
                    out_image = output_batch[i]
                    out_profile = profiles[i].copy()
                    output_file = image_out_path / filenames[i]
                    logger.debug(f"  Saving output image to: {output_file}")
                    with rio.open(output_file, 'w', **out_profile) as dst:
                        dst.write(out_image)
                
                pbar_metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
                pbar_metrics_df = pbar_metrics_df.mean().to_frame().T
                pbar.set_postfix({col: f"{pbar_metrics_df[col].values[0]:.4f}" for col in pbar_metrics_df.columns})
                
                logger.debug(f"Batch {batch_idx + 1} complete. Running averages: {pbar_metrics_df.to_dict('records')[0]}")
    
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user. Saving results obtained so far...")
        logger.debug(f"Processed {len(metrics)} samples before interruption")
    
    logger.debug(f"Creating metrics DataFrame from {len(metrics)} samples...")
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.index.name = 'sample_id'
    
    metrics_file = out_path / 'sr_evaluation_metrics.csv'
    logger.debug(f"Saving per-sample metrics to {metrics_file}...")
    metrics_df.to_csv(metrics_file)
    logger.info(f"Saved image-wise SR evaluation metrics to {metrics_file}")
    logger.debug(f"Metrics DataFrame shape: {metrics_df.shape}")
    logger.debug(f"Metrics columns: {metrics_df.columns.tolist()}")
    
    # calculate mean, median, std, variance, etc. for each metric
    logger.debug("Computing summary statistics...")
    summary_stats = metrics_df.describe().transpose()
    summary_stats.index.name = 'metric'
    
    summary_file = out_path / 'sr_evaluation_summary_stats.csv'
    logger.debug(f"Saving summary statistics to {summary_file}...")
    summary_stats.to_csv(summary_file)
    logger.info(f"Saved summary SR evaluation statistics to {summary_file}")
    logger.debug(f"Summary stats:\n{summary_stats}")
    
    logger.info('Mean SR Evaluation Metrics:' + f"\n{summary_stats['mean']}")
    
    total_end_time = time()
    total_time = total_end_time - total_start_time
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")
    logger.info(f"Total GPU sampling time: {gpu_time:.2f} seconds")
    logger.debug(f"Non-GPU time: {total_time - gpu_time:.2f} seconds ({(total_time - gpu_time) / total_time * 100:.1f}%)")
    logger.debug(f"GPU time percentage: {gpu_time / total_time * 100:.1f}%")
    logger.debug(f"Average time per sample: {total_time / len(metrics):.4f} seconds")
    logger.debug(f"Average GPU time per sample: {gpu_time / len(metrics):.4f} seconds")
    
    times_dict = {
        'total_time_seconds': total_time,
        'gpu_sampling_time_seconds': gpu_time,
        'num_samples': len(metrics),
        'avg_time_per_sample': total_time / len(metrics),
        'avg_gpu_time_per_sample': gpu_time / len(metrics)
    }
    
    times_file = out_path / 'sr_evaluation_times.json'
    logger.debug(f"Saving timing information to {times_file}...")
    with open(times_file, 'w') as f:
        json.dump(times_dict, f, indent=4)
    logger.info(f"Saved evaluation timing information to {times_file}")
    logger.debug(f"Timing dict: {times_dict}")
    
    logger.debug("SR model evaluation completed successfully")