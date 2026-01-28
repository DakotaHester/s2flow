import torch
from typing import Tuple, Union
import numpy as np

def scale(
        data: Union[torch.Tensor, np.ndarray],
        in_range: Union[Tuple[int, int], Tuple[float, float]]=(0, 10000), 
        out_range: Union[Tuple[int, int], Tuple[float, float]]=(-1.0, 1.0)
) -> torch.Tensor:
        
    # scale to 0-1
    data = (data - in_range[0]) / (in_range[1] - in_range[0])
    
    # scale to out_range
    data = data * (out_range[1] - out_range[0]) + out_range[0]
    
    # if torch tensor, clamp values
    if isinstance(data, torch.Tensor):
        data = data.clamp(min=out_range[0], max=out_range[1])
    elif isinstance(data, np.ndarray):
        data = np.clip(data, out_range[0], out_range[1])
    else:
        raise TypeError("data must be a torch.Tensor or numpy.ndarray")
    return data
