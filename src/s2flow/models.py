import torch
import torch.nn as nn
from thop import profile
from diffusers.models import UNet2DModel
from typing import Dict, Any
from logging import getLogger
import json
import os

from .utils import get_device

logger = getLogger(__name__)

def get_sr_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiate and return the model based on the provided configuration."""
    model_config = config.get("sr_model", {})
    model_type = model_config.get("model_type", "unet")
    logger.info(f"Initializing model of type: {model_type}")
    
    if model_type == 'unet':
        model = UNet2DModel(
            sample_size=model_config.get("sample_size", 256),
            in_channels=model_config.get("in_channels", 8),
            out_channels=model_config.get("out_channels", 4),
            block_out_channels=model_config.get("block_out_channels", [64, 128, 256, 512]),
            down_block_types=model_config.get("down_block_types", ["DownBlock2D"] * 4),
            up_block_types=model_config.get("up_block_types", ["UpBlock2D"] * 4),
            layers_per_block=model_config.get("layers_per_block", 2),
            norm_num_groups=model_config.get("norm_num_groups", 32),
        )
        logger.info("UNet model initialized successfully.")
    
    elif model_type == 'esrgan':
        raise NotImplementedError("ESRGAN model type is not yet implemented.")

    device = get_device()
    model.to(device)
    logger.debug(f"Model moved to device: {device}")
    
    setattr(model, 'device', device) # Attach device info to model for later use
    
    model_complexity_dict = get_model_complexity(
        model, 
        in_channels=model_config.get("in_channels", 8), 
        sample_size=model_config.get("sample_size", 256)
    )
    logger.info(f"Model parameters: {model_complexity_dict['parameters']:,}")
    logger.info(f"Model MACs: {model_complexity_dict['macs']:,}")
    logger.info(f"Model FLOPs: {model_complexity_dict['flops']:,}")
    
    log_dir = config.get('job', {}).get('logging', {}).get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'model_complexity.json'), 'w') as f:
        json.dump(model_complexity_dict, f, indent=4)
        logger.debug(f"Saved model complexity metrics to {os.path.join(log_dir, 'model_complexity.json')}")
    
    return model


def get_model_complexity(model: nn.Module, in_channels: int, sample_size: int) -> Dict[str, int]:
    """Calculate and return the model complexity metrics."""
    logger.debug("Calculating model complexity...")
    macs, params = profile(model, 
                             inputs=(torch.randn(1, in_channels, sample_size, sample_size),), 
                             verbose=False)
    flops = 2 * macs  # FLOPs is 2x MACs
    return {
        "parameters": params,
        "macs": macs,
        "flops": flops
    }