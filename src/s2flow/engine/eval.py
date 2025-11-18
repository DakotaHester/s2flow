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
    dists_metric = MultispectralDISTS(config)
    save_eval_samples = config.get('eval', {}).get('save_eval_samples', True)
    gpu_time = 0.0
    total_start_time = time()
    
    try:
        metrics = {} # structure: {sample_id: {metric_name: value, ...}, ...}
        with trange(0, len(val_samples_gdf), batch_size, desc="Evaluating SR Model") as pbar:
            for start_idx in pbar:
                end_idx = min(start_idx + batch_size, len(val_samples_gdf))
                batch_samples = val_samples_gdf.iloc[start_idx:end_idx]
                
                input_tensors = []
                target_tensors = []
                profiles = []
                filenames = []
                for _, sample in batch_samples.iterrows():
                    input_path = data_dir_path / sample['input_path']
                    target_path = data_dir_path / sample['target_path']
                    
                    with rio.open(input_path) as src:
                        input_image = src.read()
                    with rio.open(target_path) as src:
                        target_image = src.read()
                        target_profile = src.profile.copy()
                        
                    profiles.append(target_profile) # Save profile for later use
                    
                    input_tensor = scale(torch.from_numpy(input_image).float(), in_range=(0, 10000), out_range=(-1.0, 1.0))
                    target_tensor = scale(torch.from_numpy(target_image).float(), in_range=(0, 10000), out_range=(-1.0, 1.0))
                    
                    input_tensors.append(input_tensor)
                    target_tensors.append(target_tensor)
                    filenames.append(Path(input_path).name)
                
                input_batch = torch.stack(input_tensors).to(device)
                target_batch = torch.stack(target_tensors).to(device)
                
                gpu_start_time = time()
                output_batch = sampler.sample(input_batch)
                gpu_stop_time = time()
                gpu_time += (gpu_stop_time - gpu_start_time)
                
                l1_loss = F.l1_loss(output_batch, target_batch, reduction='none').mean(dim=(1, 2, 3)) # per-sample L1 loss
                psnr = TMF.image.peak_signal_noise_ratio(output_batch, target_batch, data_range=(-1, 1), reduction='none', dim=(1, 2, 3)) # per-sample PSNR
                ssim = TMF.image.structural_similarity_index_measure(output_batch, target_batch, data_range=(-1, 1), reduction='none') # per-sample SSIM
                mssim = TMF.image.multiscale_structural_similarity_index_measure(output_batch, target_batch, data_range=(-1, 1), reduction='none') # per-sample MS-SSIM
                lpips = lpips_metric(output_batch, target_batch) # per-sample LPIPS
                dists = dists_metric(output_batch, target_batch) # per-sample DISTS
                
                output_batch = scale(output_batch.cpu(), in_range=(-1.0, 1.0), out_range=(0, 10000)).numpy()
                for i in range(output_batch.shape[0]):
                    
                    sample = batch_samples.iloc[i]
                    sample_id = sample['id']
                    
                    metrics[sample_id] = {
                        'L1': l1_loss[i].item(),
                        'PSNR': psnr[i].item(),
                        'SSIM': ssim[i].item(),
                        'MS-SSIM': mssim[i].item(),
                        'LPIPS': lpips[i].item(),
                        'DISTS': dists[i].item()
                    }
                    # Save output image
                    if save_eval_samples:
                        out_image = output_batch[i]
                        out_profile = profiles[i].copy()
                        with rio.open(image_out_path / filenames[i], 'w', **out_profile) as dst:
                            dst.write(out_image)
                
                pbar_metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
                pbar_metrics_df = pbar_metrics_df.mean().to_frame().T
                pbar.set_postfix({col: f"{pbar_metrics_df[col].values[0]:.4f}" for col in pbar_metrics_df.columns})
    
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user. Saving results obtained so far...")
            
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
    
    total_end_time = time()
    total_time = total_end_time - total_start_time
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")
    logger.info(f"Total GPU sampling time: {gpu_time:.2f} seconds")
    
    times_dict = {
        'total_time_seconds': total_time,
        'gpu_sampling_time_seconds': gpu_time
    }
    with open(out_path / 'sr_evaluation_times.json', 'w') as f:
        json.dump(times_dict, f, indent=4)
    logger.info(f"Saved evaluation timing information to {out_path / 'sr_evaluation_times.json'}")
    # raise NotImplementedError("Super-resolution model evaluation is not yet implemented.")


@torch.no_grad()
def lc_model_evaluation(config: Dict[str, Any], model: nn.Module) -> None:
    """Evaluate land cover classification model on test set.
    
    Args:
        config: Configuration dictionary
        model: Land cover classification model
    """
    model.eval()
    
    # Get configuration
    samples_par_path = config.get('data', {}).get('samples_par_path', None)
    if samples_par_path is None:
        raise ValueError("samples_par_path must be specified in the config under 'data.samples_par_path'")
    
    data_dir_path = Path(config.get('data', {}).get('data_dir_path', './data/cpb_lc'))
    
    # Load test samples
    samples_gdf = gpd.read_parquet(samples_par_path)
    test_samples_gdf = samples_gdf[samples_gdf['split'] == 'test'].reset_index(drop=True)
    logger.info(f"Running inference on {len(test_samples_gdf)} test samples...")
    
    batch_size = config.get('hyperparameters', {}).get('micro_batch_size', 32)
    out_path = config['paths']['out_path']
    image_out_path = out_path / 'lc_outputs'
    image_out_path.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    num_classes = model.num_classes
    save_eval_samples = config.get('eval', {}).get('save_eval_samples', True)
    
    # Determine source data type for correct column name
    source_data = config.get('data', {}).get('source_data', 's2')
    source_col_map = {
        's2': 's2_path',
        'naip': 'naip_path',
        's2sr': 's2sr_path'
    }
    input_col_name = source_col_map.get(source_data, 's2_path')
    with rio.open(data_dir_path / test_samples_gdf.iloc[0]['lc_path']) as src:
        colormap = src.colormap(1)
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    gpu_time = 0.0
    total_start_time = time()
    
    try:
        with trange(0, len(test_samples_gdf), batch_size, desc="Evaluating LC Model") as pbar:
            for start_idx in pbar:
                end_idx = min(start_idx + batch_size, len(test_samples_gdf))
                batch_samples = test_samples_gdf.iloc[start_idx:end_idx]
                
                input_tensors = []
                target_tensors = []
                profiles = []
                filenames = []
                
                for _, sample in batch_samples.iterrows():
                    input_path = data_dir_path / sample[input_col_name]
                    target_path = data_dir_path / sample['lc_path']
                    
                    with rio.open(input_path) as src:
                        input_image = src.read()  # [C, H, W]
                    with rio.open(target_path) as src:
                        target_data = src.read(1)  # [H, W]
                        target_profile = src.profile.copy()
                    
                    profiles.append(target_profile)
                    
                    input_tensor = scale(
                        torch.from_numpy(input_image).float(), 
                        in_range=(0, 10000) if input_col_name in ('s2_path', 's2sr_path') else (0, 255),
                        out_range=(0.0, 1.0)
                    )
                    target_tensor = torch.from_numpy(target_data).long()
                    
                    input_tensors.append(input_tensor)
                    target_tensors.append(target_tensor)
                    filenames.append(Path(input_path).stem)
                
                input_batch = torch.stack(input_tensors).to(device)
                target_batch = torch.stack(target_tensors).to(device)
                
                # Forward pass
                gpu_start_time = time()
                logits_batch = model(input_batch)
                pred_batch = torch.argmax(logits_batch, dim=1)  # [B, H, W]
                gpu_stop_time = time()
                gpu_time += (gpu_stop_time - gpu_start_time)
                
                # Update confusion matrix
                for pred, target in zip(pred_batch, target_batch):
                    pred_flat = pred.cpu().flatten() + 1
                    target_flat = target.cpu().flatten()
                    
                    # Update confusion matrix
                    for t, p in zip(target_flat, pred_flat):
                        confusion_matrix[t.long(), p.long()] += 1
                
                # Save output images if requested
                if save_eval_samples:
                    pred_batch_cpu = pred_batch.cpu().numpy()
                    for i in range(pred_batch_cpu.shape[0]):
                        out_image = pred_batch_cpu[i] + 1  # Assuming LC classes start at 0, add 1 for saving
                        out_profile = profiles[i].copy()
                        out_profile.update(dtype='uint8', count=1)
                        
                        out_filename = f"{filenames[i]}_pred.tif"
                        with rio.open(image_out_path / out_filename, 'w', **out_profile) as dst:
                            dst.write(out_image, 1)
                            dst.write_colormap(1, colormap)
                
                # Compute running metrics for progress bar
                if confusion_matrix.sum() > 0:
                    overall_acc = confusion_matrix.diag().sum().float() / confusion_matrix.sum()
                    pbar.set_postfix({'accuracy': f"{overall_acc:.4f}"})
    
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user. Computing metrics from data obtained so far...")
    
    # Compute classification metrics from confusion matrix
    logger.info("Computing classification metrics from confusion matrix...")
    
    # Per-class metrics
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)
    
    # Overall accuracy
    overall_accuracy = tp.sum().float() / confusion_matrix.sum()
    
    # Per-class precision, recall, F1, IoU
    precision_per_class = tp.float() / (tp + fp).float()
    recall_per_class = tp.float() / (tp + fn).float()
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    iou_per_class = tp.float() / (tp + fp + fn).float()
    
    # Handle NaN values (divisions by zero)
    precision_per_class = torch.nan_to_num(precision_per_class, nan=0.0)
    recall_per_class = torch.nan_to_num(recall_per_class, nan=0.0)
    f1_per_class = torch.nan_to_num(f1_per_class, nan=0.0)
    iou_per_class = torch.nan_to_num(iou_per_class, nan=0.0)
    
    # Macro averages (simple mean across classes)
    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()
    macro_iou = iou_per_class.mean()  # mIoU
    
    # Micro averages (weighted by support)
    micro_precision = tp.sum().float() / (tp.sum() + fp.sum()).float()
    micro_recall = tp.sum().float() / (tp.sum() + fn.sum()).float()
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    
    # Cohen's Kappa
    po = overall_accuracy  # Observed agreement
    pe = ((confusion_matrix.sum(dim=0) * confusion_matrix.sum(dim=1)).sum().float() / 
          (confusion_matrix.sum() ** 2))  # Expected agreement
    kappa = (po - pe) / (1 - pe)
    
    # Create summary metrics dictionary
    summary_metrics = {
        'overall_accuracy': overall_accuracy.item(),
        'macro_precision': macro_precision.item(),
        'macro_recall': macro_recall.item(),
        'macro_f1': macro_f1.item(),
        'micro_precision': micro_precision.item(),
        'micro_recall': micro_recall.item(),
        'micro_f1': micro_f1.item(),
        'mean_iou': macro_iou.item(),
        'cohen_kappa': kappa.item(),
    }
    
    # Create per-class metrics dataframe
    per_class_metrics = pd.DataFrame({
        'class_id': list(range(num_classes)),
        'precision': precision_per_class.numpy(),
        'recall': recall_per_class.numpy(),
        'f1_score': f1_per_class.numpy(),
        'iou': iou_per_class.numpy(),
        'support': confusion_matrix.sum(dim=1).numpy()
    })
    
    # Save results
    logger.info("Saving evaluation results...")
    
    # Save confusion matrix
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix.numpy(),
        index=[f'true_{i}' for i in range(num_classes)],
        columns=[f'pred_{i}' for i in range(num_classes)]
    )
    confusion_matrix_df.to_csv(out_path / 'lc_confusion_matrix.csv')
    logger.info(f"Saved confusion matrix to {out_path / 'lc_confusion_matrix.csv'}")
    
    # Save per-class metrics
    per_class_metrics.to_csv(out_path / 'lc_per_class_metrics.csv', index=False)
    logger.info(f"Saved per-class metrics to {out_path / 'lc_per_class_metrics.csv'}")
    
    # Save summary metrics
    summary_metrics_df = pd.DataFrame([summary_metrics])
    summary_metrics_df.to_csv(out_path / 'lc_summary_metrics.csv', index=False)
    logger.info(f"Saved summary metrics to {out_path / 'lc_summary_metrics.csv'}")
    
    # Log summary metrics
    logger.info("Land Cover Evaluation Summary:")
    for metric_name, metric_value in summary_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Save timing information
    total_end_time = time()
    total_time = total_end_time - total_start_time
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")
    logger.info(f"Total GPU inference time: {gpu_time:.2f} seconds")
    
    times_dict = {
        'total_time_seconds': total_time,
        'gpu_inference_time_seconds': gpu_time,
        'num_samples': len(test_samples_gdf),
        'avg_time_per_sample': total_time / len(test_samples_gdf)
    }
    with open(out_path / 'lc_evaluation_times.json', 'w') as f:
        json.dump(times_dict, f, indent=4)
    logger.info(f"Saved evaluation timing information to {out_path / 'lc_evaluation_times.json'}")
