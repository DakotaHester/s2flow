from pathlib import Path
import torch
import torch.nn as nn
from thop import profile
from diffusers.models import UNet2DModel
from typing import Dict, Any, Optional, Union
from logging import getLogger
import json
from .utils import get_device

logger = getLogger(__name__)

class UNetTensorWrapper(nn.Module):
    """
    A wrapper for UNet2DModel that returns the .sample tensor 
    directly from its forward pass.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Hold the original, unmodified model
        self.model = UNet2DModel(
            sample_size=config.get("sample_size", 256),
            in_channels=config.get("in_channels", 8),
            out_channels=config.get("out_channels", 4),
            block_out_channels=config.get("block_out_channels", [64, 128, 256, 512]),
            down_block_types=config.get("down_block_types", ["DownBlock2D"] * 4),
            up_block_types=config.get("up_block_types", ["UpBlock2D"] * 4),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
        )

    def forward(
        self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # By default, return_dict=True, so this is a UNet2DOutput
        return self.model(
            sample,
            timestep,
            class_labels=class_labels
        ).sample # Return only the sample tensor (inportant!)

def get_sr_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiate and return the model based on the provided configuration."""
    model_config = config.get("sr_model", {})
    model_type = model_config.get("model_type", "unet")
    logger.info(f"Initializing model of type: {model_type}")
    
    if model_type == 'unet':
        model = UNetTensorWrapper(model_config)
        logger.info("UNet model initialized successfully.")
    
    elif model_type == 'esrgan':
        raise NotImplementedError("ESRGAN model type is not yet implemented.")

    device = get_device()
    model.to(device)
    logger.debug(f"Model moved to device: {device}")
    
    model_complexity_dict = get_model_complexity(
        model, 
        in_channels=model_config.get("in_channels", 8), 
        sample_size=model_config.get("sample_size", 256)
    )
    logger.info(f"Model parameters: {model_complexity_dict['parameters']:,}")
    logger.info(f"Model MACs: {model_complexity_dict['macs']:,}")
    logger.info(f"Model FLOPs: {model_complexity_dict['flops']:,}")
    
    log_path = config['paths']['log_path'] # raise KeyError if not found - this should be set up already
    
    with open(log_path / 'model_complexity.json', 'w') as f:
        json.dump(model_complexity_dict, f, indent=4)
        logger.debug(f"Saved model complexity metrics to {log_path / 'model_complexity.json'}")
    
    return model


def get_model_complexity(model: nn.Module, in_channels: int, sample_size: int) -> Dict[str, int]:
    """Calculate and return the model complexity metrics."""
    
    logger.debug("Calculating model complexity...")
    test_input = torch.randn(1, in_channels, sample_size, sample_size).to(next(model.parameters()).device)
    macs, params = profile(model, inputs=(test_input, 0), verbose=False) # dummy timestep verbose=False)
    flops = 2 * macs  # FLOPs is 2x MACs
    return {
        "parameters": int(params),
        "macs": int(macs),
        "flops": int(flops)
    }