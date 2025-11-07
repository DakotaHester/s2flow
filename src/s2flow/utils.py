from ctypes import Union
import torch
import logging
from time import gmtime
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

def init_logging(config: Dict[str, Any], verbose: bool=False) -> logging.Logger:
    
    logging.Formatter.converter = gmtime
    log_fmt = '[%(asctime)sZ] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%SZ'
    formatter = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)
    
    root_logger = logging.getLogger()
    if verbose:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)
        
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.debug("Initialized console logging")
    
    # log to file
    job_config = config.get("job", None)
    if job_config is None:
        raise ValueError("Job configuration must be specified in the config under 'job'")
    
    job_name = job_config.get("name", None)
    if job_name is None:
        raise ValueError("Job name must be specified in the config under 'job.name'")
    log_dir = Path(job_config.get("logging", {}).get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)\
    
    log_file = log_dir / f"{job_name}.log"
    if log_file.exists():
        root_logger.warning(f"Log file {log_file} already exists and will be overwritten.")
        log_file.unlink() 
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.debug(f"Initialized file logging at {log_file}")
    
    out_str = "Configuration:\n"
    for key, value in config.items():
        if isinstance(value, dict):
            out_str += f"{key}:\n"
            for subkey, subvalue in value.items():
                out_str += f"  {subkey}: {subvalue}\n"
        else:
            out_str += f"{key}: {value}\n"
    root_logger.info(out_str)
    
    return root_logger
    
    
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug(f'Using CUDA device: {device}')
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("cuda")
        logger.debug(f'Using MPS device: {device}')
        return device
    else:
        device = torch.device("cuda")
        logger.debug(f'Using CPU device: {device}')
        return device


def get_hp_dtype() -> torch.dtype:
    """ 
    Determine appropriate half-precision data type to use based on device 
    capabilities. Returns bfloat16 if supported, otherwise float16.

    Returns:
        torch.dtype: Half-precision data type (if supported), otherwise float32.
    """
    if torch.cuda.is_bf16_supported():
        logger.debug('utils.get_hp_dtype: Using bfloat16 dtype for CUDA device with bf16 support')
        return torch.bfloat16
    else: # have cuda but no bf16
        logger.debug(f'utils.get_hp_dtype: Using float16 dtype for device without bf16 support')
        return torch.float16