import torch.nn as nn
from diffusers.models import UNet2DModel

from typing import Dict, Union

def get_model(config: Dict[str, Union[str, int, float]]) -> nn.Module:
    """Instantiate and return the model based on the provided configuration."""
    model_type = config.get("model_type", "unet")
    
    if model_type == 'unet':
        model = UNet2DModel(
            sample_size=config.get("sample_size", 256),
            in_channels=config.get("in_channels", 8),
            out_channels=config.get("out_channels", 4),
            block_out_channels=config.get("block_out_channels", [64, 128, 256, 512]),
            down_block_types=config.get("down_block_types", ["DownBlock2D"] * 4),
            up_block_types=config.get("up_block_types", ["UpBlock2D"] * 4),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
        )
    
    elif model_type == 'esrgan':
        raise NotImplementedError("ESRGAN model type is not yet implemented.")
    
    return model