import torch
import torch.nn as nn
from typing import Any, Dict
from torch.cuda.amp import autocast
from tqdm import tqdm

from ..utils import get_hp_dtype, get_device


class BaseSampler:
    def __init__(self, config: Dict[str, Any], model: nn.Module) -> None:
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        
        self.num_timesteps = config.get('sampling', {}).get('num_timesteps', 50)
    
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Sampler must implement the sample method.")


class EulerSampler(BaseSampler):
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        return euler_sampling(self.model, cond, self.num_timesteps)


class RK4Sampler(BaseSampler):
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        return rk4_sampling(self.model, cond, self.num_timesteps)


def euler_sampling(model: nn.Module, cond: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    
    device = model.device
    hp_dtype = get_hp_dtype() # Use half-precision if supported
    timesteps = torch.linspace(0, 1, num_timesteps, device=device)
    step_size = timesteps[1] - timesteps[0]
    
    x = torch.randn(cond.size(), device=device)
    for t in timesteps:
        t_batch = torch.ones(cond.shape[0], device=device) * t
        
        model_input = torch.cat((x, cond), dim=1)
        with autocast(device_type=device.type, dtype=hp_dtype, enabled=hp_dtype != torch.float32), torch.no_grad():
            v = model(model_input, t_batch * 1000).sample
        
        x = x + v * step_size
    
    return x


def rk4_sampling(model: nn.Module, cond: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    
    device = model.device
    hp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    timesteps = torch.linspace(0, 1, num_timesteps, device=device)
    step_size = timesteps[1] - timesteps[0]
    
    naip_shape = list(cond.shape)
    naip_shape[1] = 4
    x = torch.randn(naip_shape, device=device)
    for t in tqdm(timesteps):
        t_batch = torch.ones(cond.shape[0], device=device) * t
        t_mid_batch = torch.ones(cond.shape[0], device=device) * (t + step_size / 2)
        t_next_batch = torch.ones(cond.shape[0], device=device) * (t + step_size)
            
        # k1: velocity at start of the step
        model_input_k1 = torch.cat((x, cond), dim=1)
        with autocast(device_type=device.type, dtype=hp_dtype):
            v_k1 = model(model_input_k1, t_batch * 1000).sample
        
        # k2: velocity at midpoint
        x_k2 = x + v_k1 * (step_size / 2)
        model_input_k2 = torch.cat((x_k2, cond), dim=1)
        with autocast(device_type=device.type, dtype=hp_dtype):
            v_k2 = model(model_input_k2, t_mid_batch * 1000).sample
        
        # k3: velocity at midpoint (estimated with k2)
        x_k3 = x + v_k2 * (step_size / 2)
        model_input_k3 = torch.cat((x_k3, cond), dim=1)
        with autocast(device_type=device.type, dtype=hp_dtype):
            v_k3 = model(model_input_k3, t_mid_batch * 1000).sample
        
        # k4: velocity at the end of the step
        x_k4 = x + v_k3 * step_size
        model_input_k4 = torch.cat((x_k4, cond), dim=1)
        with autocast(device_type=device.type, dtype=hp_dtype):
            v_k4 = model(model_input_k4, t_next_batch * 1000).sample
        
        x = x + (step_size / 6) * (v_k1 + 2*v_k2 + 2*v_k3 + v_k4)
    
    return x


def get_sampler(config: Dict[str, Any], model: nn.Module) -> BaseSampler:
    sampler_type = config.get('sampling', {}).get('sampler_type', 'euler').lower()
    
    if sampler_type == 'euler':
        sampler = EulerSampler(config, model)
    elif sampler_type == 'rk4':
        sampler = RK4Sampler(config, model)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")
    
    return sampler