import torch
from typing import Tuple, Union


def scale(
        data, 
        in_range: Union[Tuple[int, int], Tuple[float, float]]=(0, 10000), 
        out_range: Union[Tuple[int, int], Tuple[float, float]]=(-1.0, 1.0)
) -> torch.Tensor:
        
    # scale to 0-1
    data = (data - in_range[0]) / (in_range[1] - in_range[0])
    
    # scale to out_range
    data = data * (out_range[1] - out_range[0]) + out_range[0]
    data = data.clamp(min=out_range[0], max=out_range[1])
    return data
