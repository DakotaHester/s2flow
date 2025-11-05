import argparse
import os
from typing import Any, Dict
import yaml
import logging
from .data.datasets import get_dataloaders
from .utils import init_logging

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Super-Resolution Model using Flow Matching")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output.'
    )
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger = init_logging(config, verbose=args.verbose)
    
    job_type = config.get("job", {}).get("type", None)
    if job_type is None:
        raise ValueError("Job type must be specified in the config under 'job.type'")
    
    elif job_type == 'sr_train':
        from s2flow.engine.training import train_sr_model
        train_sr_model(config)
        
    elif job_type == 'sr_eval':
        raise NotImplementedError("Super-resolution model evaluation is not yet implemented.")
    
    elif job_type == 'sr_inference':
        sr_model_inference(config, logger)
        
    elif job_type == 'lc_train':
        train_lc_model(config, logger)
        
    elif job_type == 'lc_eval':
        eval_lc_model(config, logger)
        
    elif job_type == 'lc_inference':
        lc_model_inference(config, logger)
        
    else:
        raise ValueError(
            f"Unknown job type: {job_type}. Must be one of 'sr_train'," + \
            "'sr_eval', 'sr_inference', 'lc_train', 'lc_eval', 'lc_inference'."
        )

def sr_model_inference(config: Dict[str, Any], logger: logging.Logger):
    raise NotImplementedError("Super-resolution model inference is not yet implemented.")
    

def train_lc_model(config: Dict[str, Any], logger: logging.Logger):
    raise NotImplementedError("Land cover model training is not yet implemented.")


def eval_lc_model(config: Dict[str, Any], logger: logging.Logger):
    raise NotImplementedError("Land cover model evaluation is not yet implemented.")


def lc_model_inference(config: Dict[str, Any], logger: logging.Logger):
    raise NotImplementedError("Land cover model inference is not yet implemented.")


if __name__ == "__main__":
    main()