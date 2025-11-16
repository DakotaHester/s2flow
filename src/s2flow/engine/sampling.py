import torch
import torch.nn as nn
from typing import Any, Dict
from torch.amp import autocast
from tqdm import tqdm
from logging import getLogger
from contextlib import nullcontext
from abc import ABC, abstractmethod, ABCMeta

from ..utils import get_hp_dtype, get_device
logger = getLogger(__name__)

class BaseSampler(ABC):
    def __init__(self, config: Dict[str, Any], model: nn.Module) -> None:
        __metaclass__ = ABCMeta
        
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        
        self.use_amp = config.get('hyperparameters', None).get('use_amp', True)
        if self.use_amp:
            hp_dtype = get_hp_dtype()
            logger.debug(f"Using AMP with dtype: {hp_dtype}")
            self.autocast_context = autocast(device_type=self.device.type, dtype=hp_dtype, enabled=True)
        else:
            logger.debug("AMP disabled; using full precision (float32).")
            self.autocast_context = nullcontext()
        
        self.num_timesteps = config.get('sampling', {}).get('num_steps', 50)
        self.timesteps = torch.linspace(0, 1, self.num_timesteps, device=self.device)
        if self.num_timesteps > 1:
            self.step_size = self.timesteps[1] - self.timesteps[0]
        elif self.num_timesteps == 1:
            self.step_size = 1.0
        else:
            raise ValueError("num_steps must be at least 1.")
    
    @abstractmethod
    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        pass


class BaseSampler(ABC):
    def __init__(self, config: Dict[str, Any], model: nn.Module) -> None:
        __metaclass__ = ABCMeta
        
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        
        self.use_amp = config.get('hyperparameters', None).get('use_amp', True)
        if self.use_amp:
            hp_dtype = get_hp_dtype()
            logger.debug(f"Using AMP with dtype: {hp_dtype}")
            self.autocast_context = autocast(device_type=self.device.type, dtype=hp_dtype, enabled=True)
        else:
            logger.debug("AMP disabled; using full precision (float32).")
            self.autocast_context = nullcontext()
        
        self.num_timesteps = config.get('sampling', {}).get('num_steps', 50.0)
        if self.num_timesteps > 0:
            self.step_size = 1 / self.num_timesteps
            self.timesteps = torch.linspace(0.0, 1 - self.step_size, self.num_timesteps, device=self.device)
        else:
            raise ValueError(f"num_timesteps must be at least 1, got {self.num_timesteps}.")
    
    @abstractmethod
    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        pass


class EulerSampler(BaseSampler):
    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        
        x = torch.randn_like(cond, device=self.device)
        for t in tqdm(self.timesteps, desc="Sampling", leave=False, unit="step"):
            t_batch = torch.ones(cond.shape[0], device=self.device) * t
            
            model_input = torch.cat((x, cond), dim=1)
            with self.autocast_context:
                v = self.model(model_input, t_batch * 1000)
            
            x = x + v * self.step_size
        
        return x


class HeunSampler(BaseSampler):
    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:

        x = torch.randn_like(cond, device=self.device)
        for t in tqdm(self.timesteps, desc="Sampling", leave=False, unit="step"):
            t_batch = torch.ones(cond.shape[0], device=self.device) * t
            t_next_batch = torch.ones(cond.shape[0], device=self.device) * (t + self.step_size)
            
            # Predictor step
            model_input = torch.cat((x, cond), dim=1)
            with self.autocast_context:
                v1 = self.model(model_input, t_batch * 1000)
            x_pred = x + v1 * self.step_size
            
            # Corrector step
            model_input_pred = torch.cat((x_pred, cond), dim=1)
            with self.autocast_context:
                v2 = self.model(model_input_pred, t_next_batch * 1000)
            
            x = x + (self.step_size / 2) * (v1 + v2)
        
        return x


class MidpointSampler(BaseSampler):
    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:

        x = torch.randn_like(cond, device=self.device) 
        for t in tqdm(self.timesteps, desc="Sampling", leave=False, unit="step"):
            t_batch = torch.ones(cond.shape[0], device=self.device) * t
            t_mid_batch = torch.ones(cond.shape[0], device=self.device) * (t + (self.step_size / 2))
            
            # Compute velocity at the start of the step
            model_input = torch.cat((x, cond), dim=1)
            with self.autocast_context:
                v1 = self.model(model_input, t_batch * 1000)
            
            # Estimate midpoint
            x_mid = x + v1 * (self.step_size / 2)
            model_input_mid = torch.cat((x_mid, cond), dim=1)
            with self.autocast_context:
                v2 = self.model(model_input_mid, t_mid_batch * 1000)
            
            x = x + v2 * self.step_size
        
        return x


class RK4Sampler(BaseSampler):
    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:

        x = torch.randn_like(cond, device=self.device)
        for t in tqdm(self.timesteps, desc="Sampling", leave=False, unit="step"):
            t_batch = torch.ones(cond.shape[0], device=self.device) * t
            t_mid_batch = torch.ones(cond.shape[0], device=self.device) * (t + (self.step_size / 2))
            t_next_batch = torch.ones(cond.shape[0], device=self.device) * (t + self.step_size)
            
            # k1: velocity at start of the step
            model_input_k1 = torch.cat((x, cond), dim=1)
            with self.autocast_context:
                v_k1 = self.model(model_input_k1, t_batch * 1000)
            
            # k2: velocity at midpoint
            x_k2 = x + v_k1 * (self.step_size / 2)
            model_input_k2 = torch.cat((x_k2, cond), dim=1)
            with self.autocast_context:
                v_k2 = self.model(model_input_k2, t_mid_batch * 1000)
            
            # k3: velocity at midpoint (estimated with k2)
            x_k3 = x + v_k2 * (self.step_size / 2)
            model_input_k3 = torch.cat((x_k3, cond), dim=1)
            with self.autocast_context:
                v_k3 = self.model(model_input_k3, t_mid_batch * 1000)
                
            # k4: velocity at the end of the step
            x_k4 = x + v_k3 * self.step_size
            model_input_k4 = torch.cat((x_k4, cond), dim=1)
            with self.autocast_context:
                v_k4 = self.model(model_input_k4, t_next_batch * 1000)
            
            x = x + (self.step_size / 6) * (v_k1 + 2*v_k2 + 2*v_k3 + v_k4)
        
        return x



def get_sampler(config: Dict[str, Any], model: nn.Module) -> BaseSampler:
    sampler_type = config.get('sampling', {}).get('solver', 'euler').lower()
    
    if sampler_type == 'euler':
        logger.debug("Using Euler solver.")
        sampler = EulerSampler(config, model)
    elif sampler_type == 'heun':
        logger.debug("Using Heun solver.")
        sampler = HeunSampler(config, model)
    elif sampler_type == 'midpoint':
        logger.debug("Using Midpoint solver.")
        sampler = MidpointSampler(config, model)
    elif sampler_type == 'rk4':
        logger.debug("Using RK4 solver.")
        sampler = RK4Sampler(config, model)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")
    
    return sampler