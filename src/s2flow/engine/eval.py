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
from contextlib import nullcontext

from ..utils import get_device, get_hp_dtype
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
    """Evaluate land cover classification model on test set."""
    logger.debug("Starting lc_model_evaluation")
    model.eval()
    
    samples_par_path = config.get('data', {}).get('samples_par_path', None)
    logger.debug(f"samples_par_path: {samples_par_path}")
    if samples_par_path is None:
        raise ValueError("samples_par_path must be specified in the config under 'data.samples_par_path'")
    
    data_dir_path = Path(config.get('data', {}).get('data_dir_path', './data/cpb_lc'))
    logger.debug(f"data_dir_path: {data_dir_path}")
    
    samples_gdf = gpd.read_parquet(samples_par_path)
    logger.debug(f"Loaded samples_gdf with shape: {samples_gdf.shape}")
    test_samples_gdf = samples_gdf[samples_gdf['split'] == 'test'].reset_index(drop=True)
    # TEMPORARY: Use 'val' split for testing
    # test_samples_gdf = samples_gdf[samples_gdf['split'] == 'train'].reset_index(drop=True)
    # test_samples_gdf = test_samples_gdf[test_samples_gdf['fold'] == config.get('data', {}).get('fold', 0)].reset_index(drop=True)
    logger.debug(f"Filtered test_samples_gdf with shape: {test_samples_gdf.shape}")
    logger.info(f"Running inference on {len(test_samples_gdf)} test samples...")
    
    batch_size = config.get('hyperparameters', {}).get('micro_batch_size', 32)
    logger.debug(f"batch_size: {batch_size}")
    out_path = config['paths']['out_path']
    logger.debug(f"out_path: {out_path}")
    image_out_path = out_path / 'lc_outputs'
    image_out_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"image_out_path: {image_out_path}")
    
    device = get_device()
    logger.debug(f"device: {device}")
    num_classes = config.get('lc_model', {}).get('num_classes', 7)
    logger.debug(f"num_classes: {num_classes}")
    save_eval_samples = config.get('eval', {}).get('save_eval_samples', True)
    logger.debug(f"save_eval_samples: {save_eval_samples}")
    
    source_data = config.get('data', {}).get('source_data', 's2')
    logger.debug(f"source_data: {source_data}")
    source_col_map = {
        's2': 's2_path',
        'naip': 'naip_path',
        's2sr': 's2sr_path'
    }
    input_col_name = source_col_map.get(source_data, 's2_path')
    logger.debug(f"input_col_name: {input_col_name}")
    colormap = None
    try:
        with rio.open(data_dir_path / test_samples_gdf.iloc[0]['lc_path']) as src:
            colormap = src.colormap(1)
        logger.debug("Loaded colormap from first LC file")
    except Exception as e:
        logger.warning(f"Could not load colormap from LC file: {e}. Proceeding without colormap.")
    
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    logger.debug(f"Initialized confusion_matrix: {confusion_matrix.shape}")
    
    gpu_time = 0.0
    total_start_time = time()
    if config.get('hyperparameters', {}).get('use_amp', True):
        auto_cast_ctx = torch.amp.autocast(device_type=device.type, dtype=get_hp_dtype())
        logger.info(f"Using AMP with dtype: {get_hp_dtype()}")
    else:
        auto_cast_ctx = nullcontext()
        logger.info("Not using AMP for evaluation.")
    logger.debug("Starting evaluation loop")
    
    try:
        with trange(0, len(test_samples_gdf), batch_size, desc="Evaluating LC Model") as pbar:
            for start_idx in pbar:
                logger.debug(f"Batch start_idx: {start_idx}")
                end_idx = min(start_idx + batch_size, len(test_samples_gdf))
                logger.debug(f"Batch end_idx: {end_idx}")
                batch_samples = test_samples_gdf.iloc[start_idx:end_idx]
                logger.debug(f"Batch samples shape: {batch_samples.shape}")
                
                input_tensors = []
                target_tensors = []
                profiles = []
                filenames = []
                
                for row_idx, sample in batch_samples.iterrows():
                    logger.debug(f"Processing sample index: {row_idx}, id: {sample.get('id', 'N/A')}")
                    input_path = data_dir_path / sample[input_col_name]
                    target_path = data_dir_path / sample['lc_path']
                    logger.debug(f"input_path: {input_path}, target_path: {target_path}")
                    
                    with rio.open(input_path) as src:
                        input_image = src.read()
                        logger.debug(f"Read input_image shape: {input_image.shape}")
                    with rio.open(target_path) as src:
                        target_data = src.read(1)
                        target_profile = src.profile.copy()
                        logger.debug(f"Read target_data shape: {target_data.shape}")
                    
                    profiles.append(target_profile)
                    
                    input_tensor = scale(
                        torch.from_numpy(input_image).float(), 
                        in_range=(0, 10000) if input_col_name in ('s2_path', 's2sr_path') else (0, 255),
                        out_range=(0.0, 1.0)
                    )
                    logger.debug(f"Scaled input_tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
                    target_tensor = torch.from_numpy(target_data).long() - 1
                    logger.debug(f"Processed target_tensor shape: {target_tensor.shape}, dtype: {target_tensor.dtype}")
                    
                    input_tensors.append(input_tensor)
                    target_tensors.append(target_tensor)
                    filenames.append(Path(input_path).stem)
                
                input_batch = torch.stack(input_tensors).to(device)
                target_batch = torch.stack(target_tensors).to(device)
                logger.debug(f"input_batch shape: {input_batch.shape}, target_batch shape: {target_batch.shape}")
                
                gpu_start_time = time()
                with auto_cast_ctx:
                    logits_batch = model(input_batch)
                logger.debug(f"logits_batch shape: {logits_batch.shape}")
                pred_batch = torch.argmax(logits_batch, dim=1)
                logger.debug(f"pred_batch shape: {pred_batch.shape}")
                gpu_stop_time = time()
                gpu_time += (gpu_stop_time - gpu_start_time)
                logger.debug(f"Batch GPU time: {gpu_stop_time - gpu_start_time:.4f}s, Total GPU time: {gpu_time:.4f}s")
                
                for pred, target in zip(pred_batch, target_batch):
                    pred_flat = pred.cpu().flatten()
                    target_flat = target.cpu().flatten()
                    logger.debug(f"Updating confusion matrix for batch sample, pred_flat shape: {pred_flat.shape}, target_flat shape: {target_flat.shape}")
                    for t, p in zip(target_flat, pred_flat):
                        confusion_matrix[t.long(), p.long()] += 1
                logger.debug(f"Updated confusion_matrix sum: {confusion_matrix.sum().item()}")
                
                if save_eval_samples:
                    pred_batch_cpu = pred_batch.cpu().numpy()
                    logger.debug(f"Saving predicted images, batch size: {pred_batch_cpu.shape[0]}")
                    for i in range(pred_batch_cpu.shape[0]):
                        out_image = pred_batch_cpu[i] + 1
                        out_profile = profiles[i].copy()
                        out_profile.update(dtype='uint8', count=1, photometric='palette')
                        out_filename = f"{filenames[i]}.tif"
                        logger.debug(f"Saving output image: {out_filename}")
                        with rio.open(image_out_path / out_filename, 'w', **out_profile) as dst:
                            dst.write(out_image, 1)
                            if colormap is not None:
                                dst.write_colormap(1, colormap)
                
                if confusion_matrix.sum() > 0:
                    overall_acc = confusion_matrix.diag().sum().float() / confusion_matrix.sum()
                    logger.debug(f"Current overall accuracy: {overall_acc:.4f}")
                    pbar.set_postfix({'accuracy': f"{overall_acc:.4f}"})
    
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user. Computing metrics from data obtained so far...")
    
    logger.info("Computing classification metrics from confusion matrix...")
    logger.debug(f"Final confusion_matrix:\n{confusion_matrix}")
    
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)
    logger.debug(f"tp: {tp}")
    logger.debug(f"fp: {fp}")
    logger.debug(f"fn: {fn}")
    logger.debug(f"tn: {tn}")
    
    overall_accuracy = tp.sum().float() / confusion_matrix.sum()
    logger.debug(f"overall_accuracy: {overall_accuracy:.4f}")
    
    precision_per_class = tp.float() / (tp + fp).float()
    recall_per_class = tp.float() / (tp + fn).float()
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    iou_per_class = tp.float() / (tp + fp + fn).float()
    logger.debug(f"precision_per_class: {precision_per_class}")
    logger.debug(f"recall_per_class: {recall_per_class}")
    logger.debug(f"f1_per_class: {f1_per_class}")
    logger.debug(f"iou_per_class: {iou_per_class}")
    
    precision_per_class = torch.nan_to_num(precision_per_class, nan=0.0)
    recall_per_class = torch.nan_to_num(recall_per_class, nan=0.0)
    f1_per_class = torch.nan_to_num(f1_per_class, nan=0.0)
    iou_per_class = torch.nan_to_num(iou_per_class, nan=0.0)
    logger.debug("NaN values replaced in per-class metrics")
    
    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()
    macro_iou = iou_per_class.mean()
    logger.debug(f"macro_precision: {macro_precision:.4f}, macro_recall: {macro_recall:.4f}, macro_f1: {macro_f1:.4f}, macro_iou: {macro_iou:.4f}")
    
    micro_precision = tp.sum().float() / (tp.sum() + fp.sum()).float()
    micro_recall = tp.sum().float() / (tp.sum() + fn.sum()).float()
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    logger.debug(f"micro_precision: {micro_precision:.4f}, micro_recall: {micro_recall:.4f}, micro_f1: {micro_f1:.4f}")
    
    po = overall_accuracy
    pe = ((confusion_matrix.sum(dim=0) * confusion_matrix.sum(dim=1)).sum().float() / 
          (confusion_matrix.sum() ** 2))
    kappa = (po - pe) / (1 - pe)
    logger.debug(f"cohen_kappa: {kappa:.4f}")
    
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
    logger.debug(f"summary_metrics: {summary_metrics}")
    
    per_class_metrics = pd.DataFrame({
        'class_id': list(range(1, num_classes+1)),
        'precision': precision_per_class.numpy(),
        'recall': recall_per_class.numpy(),
        'f1_score': f1_per_class.numpy(),
        'iou': iou_per_class.numpy(),
        'support': confusion_matrix.sum(dim=1).numpy()
    })
    logger.debug(f"per_class_metrics dataframe:\n{per_class_metrics}")
    
    logger.info("Saving evaluation results...")
    
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix.numpy(),
        index=[f'true_{i}' for i in range(1, num_classes+1)],
        columns=[f'pred_{i}' for i in range(1, num_classes+1)]
    )
    confusion_matrix_df.to_csv(out_path / 'lc_confusion_matrix.csv')
    logger.debug(f"Saved confusion matrix to {out_path / 'lc_confusion_matrix.csv'}")
    
    per_class_metrics.to_csv(out_path / 'lc_per_class_metrics.csv', index=False)
    logger.debug(f"Saved per-class metrics to {out_path / 'lc_per_class_metrics.csv'}")
    
    summary_metrics_df = pd.DataFrame([summary_metrics])
    summary_metrics_df.to_csv(out_path / 'lc_summary_metrics.csv', index=False)
    logger.debug(f"Saved summary metrics to {out_path / 'lc_summary_metrics.csv'}")
    
    logger.info("Land Cover Evaluation Summary:")
    for metric_name, metric_value in summary_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    total_end_time = time()
    total_time = total_end_time - total_start_time
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")
    logger.info(f"Total GPU inference time: {gpu_time:.2f} seconds")
    logger.debug(f"Timing info: total_time={total_time:.4f}, gpu_time={gpu_time:.4f}")
    
    times_dict = {
        'total_time_seconds': total_time,
        'gpu_inference_time_seconds': gpu_time,
        'num_samples': len(test_samples_gdf),
        'avg_time_per_sample': total_time / len(test_samples_gdf)
    }
    with open(out_path / 'lc_evaluation_times.json', 'w') as f:
        json.dump(times_dict, f, indent=4)
    logger.debug(f"Saved evaluation timing information to {out_path / 'lc_evaluation_times.json'}")
