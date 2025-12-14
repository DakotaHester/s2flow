import argparse
from pathlib import Path
from shutil import copy2
from typing import Any, Dict
import yaml
import logging
import torch
from .data.datasets import get_dataloaders
from .utils import init_logging, get_device

torch.manual_seed(1701)
torch.cuda.manual_seed_all(1701)
torch.backends.cudnn.benchmark = True


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
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {args.config}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.backends.cudnn.deterministic = config.get('job', {}).get('cudnn_deterministic', False)
    
    # configure paths
    job_name = config.get('job', {}).get('name', None)
    if job_name is None:
        raise ValueError("Job name must be specified in the config under 'job.name'")
    
    log_path = Path(config.get('job', {}).get('log_dir', './logs'))
    log_path = log_path / job_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    # copy config file to log directory for reference
    copy2(config_path, log_path / 'config.yaml')
    
    out_path = Path(config.get('job', {}).get('out_dir', './runs'))
    out_path = out_path / job_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    config['paths'] = {
        'log_path': log_path,
        'out_path': out_path
    }
    logger = init_logging(config, verbose=args.verbose)
    logger.info(f'Device: {get_device()}')
    
    job_type = config.get("job", {}).get("type", None)
    if job_type is None:
        raise ValueError("Job type must be specified in the config under 'job.type'")
    
    elif job_type == 'sr_train':
        train_sr_model(config, logger)
        
    elif job_type == 'sr_eval':
        eval_sr_model(config, logger)
    
    elif job_type == 'sr_inference':
        sr_model_inference(config, logger)
    
    elif job_type == 'sr_sliding_window':
        sr_sliding_window_inference(config, logger)
        
    elif job_type == 'lc_train':
        train_lc_model(config, logger)
        
    elif job_type == 'lc_eval':
        eval_lc_model(config, logger)
        
    elif job_type == 'lc_inference':
        lc_model_inference(config, logger)
    
    elif job_type == 'lc_sliding_window':
        lc_sliding_window_inference(config, logger)
        
    else:
        raise ValueError(
            f"Unknown job type: {job_type}. Must be one of 'sr_train', 'sr_eval', " + \
            "'sr_inference', 'sr_sliding_window', 'lc_train', 'lc_eval', " + \
            "'lc_inference', 'lc_sliding_window'."
        )
    
    if config.get('job', {}).get('add_completed_file', True):
        completed_file_path = out_path / 'COMPLETE'
        completed_file_path.touch()
        logger.info(f"Created completed file at {completed_file_path}")


def train_sr_model(config: Dict[str, Any], logger: logging.Logger):
    from .engine.training import FlowMatchingSRTrainer
    from .engine.eval import sr_model_evaluation
    from .models import get_sr_model
    from .utils import get_device
    
    model = get_sr_model(config)
    pretrained_weights = config.get('sr_model', {}).get('pretrained_weights', None)
    if pretrained_weights is not None:
        logger.info(f"Loading pretrained weights from {pretrained_weights}...")
        model.load_state_dict(torch.load(pretrained_weights, map_location=get_device(), weights_only=True))
        logger.info("Pretrained weights loaded successfully.")
    
    load_checkpoint = config.get('job', {}).get('load_checkpoint', False)
    if load_checkpoint:
        logger.info("`load_checkpoint` is True; loading trainer from checkpoint...")
        trainer = FlowMatchingSRTrainer.from_checkpoint(config, model)
    else:
        logger.info("`load_checkpoint` is False; initializing new trainer...")
        trainer = FlowMatchingSRTrainer(config, model)
    
    logger.info("Setting up data loaders...")
    train_loader, val_loader = get_dataloaders(config)
    
    logger.info("Starting super-resolution model training...")
    trainer.fit(train_loader, val_loader)
    logger.info("Super-resolution model training complete.")
    
    logger.info("Starting super-resolution model evaluation...")
    sr_model_evaluation(config, model)
    logger.info("Super-resolution model evaluation complete.")


@torch.no_grad()
def eval_sr_model(config: Dict[str, Any], logger: logging.Logger):
    from .engine.eval import sr_model_evaluation
    from .models import get_sr_model
    from .utils import get_device
    
    model = get_sr_model(config)
    logger.info("Loading pretrained weights for evaluation...")
    weights_path = config.get('sr_model', {}).get('pretrained_weights', None)
    
    if weights_path is None:
        raise ValueError("Pretrained weights path must be specified in the config under 'sr_model.pretrained_weights' when running evaluation jobs.")
    
    weights = torch.load(weights_path, map_location=get_device(), weights_only=True)
    model.load_state_dict(weights, strict=False)
    logger.info("Pretrained weights loaded successfully.")

    sr_model_evaluation(config, model)
    

def sr_model_inference(config: Dict[str, Any], logger: logging.Logger):
    from .engine.inference import simple_sr_model_inference
    from .models import get_sr_model
    from .utils import get_device
    
    model = get_sr_model(config)
    logger.info("Loading pretrained weights for inference...")
    weights_path = config.get('sr_model', {}).get('pretrained_weights', None)
    
    if weights_path is None:
        raise ValueError("Pretrained weights path must be specified in the config under 'sr_model.pretrained_weights' when running inference jobs.")
    
    weights = torch.load(weights_path, map_location=get_device(), weights_only=True)
    model.load_state_dict(weights, strict=False)
    logger.info("Pretrained weights loaded successfully.")

    simple_sr_model_inference(config, model)


@torch.no_grad()
def sr_sliding_window_inference(config: Dict[str, Any], logger: logging.Logger):
    """Run sliding window super-resolution inference on a Sentinel-2 tile."""
    from .engine.sliding_window import SRSlidingWindowProcessor
    from .models import get_sr_model
    from .utils import get_device
    
    # Load SR model
    model = get_sr_model(config)
    logger.info("Loading SR pretrained weights for sliding window inference...")
    weights_path = config.get('sr_model', {}).get('pretrained_weights', None)
    
    if weights_path is None:
        raise ValueError(
            "SR pretrained weights path must be specified in the config under "
            "'sr_model.pretrained_weights' when running sliding window inference."
        )
    
    weights = torch.load(weights_path, map_location=get_device(), weights_only=True)
    model.load_state_dict(weights, strict=False)
    logger.info("SR pretrained weights loaded successfully.")
    
    # Get paths from config
    input_path = Path(config.get('data', {}).get('input_path'))
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path = config.get('data', {}).get('output_path', None)
    if output_path is None:
        output_dir = config['paths']['out_path']
        output_path = output_dir / f"{input_path.stem}_sr.tif"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    logger.info("Initializing SR sliding window processor...")
    processor = SRSlidingWindowProcessor(config, model)
    
    # Determine output path
    input_stem = input_path.stem
    output_path = output_dir / f"{input_stem}_sr.tif"
    
    # Process file
    logger.info(f"Processing input file: {input_path}")
    logger.info(f"Output will be saved to: {output_path}")
    processor.process_file(input_path=input_path, output_path=output_path)
    
    logger.info("Sliding window SR inference complete.")
    

def train_lc_model(config: Dict[str, Any], logger: logging.Logger):
    from .engine.training import LandCoverTrainer
    from .engine.eval import lc_model_evaluation
    from .models import get_lc_model
    from .utils import get_device
    
    model = get_lc_model(config)
    pretrained_weights = config.get('lc_model', {}).get('pretrained_weights', None)
    if pretrained_weights is not None:
        logger.info(f"Loading pretrained weights from {pretrained_weights}...")
        model.load_state_dict(torch.load(pretrained_weights, map_location=get_device(), weights_only=True))
        logger.info("Pretrained weights loaded successfully.")
    
    load_checkpoint = config.get('job', {}).get('load_checkpoint', False)
    if load_checkpoint:
        logger.info("`load_checkpoint` is True; loading trainer from checkpoint...")
        trainer = LandCoverTrainer.from_checkpoint(config, model)
    else:
        logger.info("`load_checkpoint` is False; initializing new trainer...")
        trainer = LandCoverTrainer(config, model)
    
    logger.info("Setting up data loaders...")
    train_loader, val_loader = get_dataloaders(config)
    
    logger.info("Starting land cover model training...")
    trainer.fit(train_loader, val_loader)
    logger.info("Land cover model training complete.")
    
    logger.info("Starting land cover model evaluation...")
    lc_model_evaluation(config, model)
    logger.info("Land cover model evaluation complete.")


def eval_lc_model(config: Dict[str, Any], logger: logging.Logger):
    from .engine.eval import lc_model_evaluation
    from .models import get_lc_model
    from .utils import get_device
    
    model = get_lc_model(config)
    logger.info("Loading pretrained weights for evaluation...")
    weights_path = config.get('lc_model', {}).get('pretrained_weights', None)
    
    if weights_path is None:
        raise ValueError("Pretrained weights path must be specified in the config under 'lc_model.pretrained_weights' when running evaluation jobs.")
    
    weights = torch.load(weights_path, map_location=get_device(), weights_only=True)
    model.load_state_dict(weights, strict=True)
    logger.info("Pretrained weights loaded successfully.")

    logger.info(f'MODEL DEVICE, {next(model.parameters()).device}')
    lc_model_evaluation(config, model)


@torch.no_grad()
def lc_model_inference(config: Dict[str, Any], logger: logging.Logger):
    """Run simple land cover model inference on a directory of images."""
    from .engine.inference import simple_lc_model_inference
    from .models import get_lc_model
    from .utils import get_device
    
    # Load LC model
    model = get_lc_model(config)
    logger.info("Loading LC pretrained weights for inference...")
    weights_path = config.get('lc_model', {}).get('pretrained_weights', None)
    
    if weights_path is None:
        raise ValueError(
            "LC pretrained weights path must be specified in the config under "
            "'lc_model.pretrained_weights' when running inference jobs."
        )
    
    weights = torch.load(weights_path, map_location=get_device(), weights_only=True)
    model.load_state_dict(weights, strict=True)
    logger.info("LC pretrained weights loaded successfully.")
    
    # Run inference
    simple_lc_model_inference(config, model)


@torch.no_grad()
def lc_sliding_window_inference(config: Dict[str, Any], logger: logging.Logger):
    """Run sliding window SR + land cover inference on a Sentinel-2 tile."""
    from .engine.sliding_window import LCSlidingWindowProcessor
    from .models import get_sr_model, get_lc_model
    from .utils import get_device
    
    device = get_device()
    
    # Load SR model
    sr_model = get_sr_model(config)
    logger.info("Loading SR pretrained weights for sliding window inference...")
    sr_weights_path = config.get('sr_model', {}).get('pretrained_weights', None)
    
    if sr_weights_path is None:
        raise ValueError(
            "SR pretrained weights path must be specified in the config under "
            "'sr_model.pretrained_weights' when running LC sliding window inference."
        )
    
    sr_weights = torch.load(sr_weights_path, map_location=device, weights_only=True)
    sr_model.load_state_dict(sr_weights, strict=False)
    sr_model.eval()
    logger.info("SR pretrained weights loaded successfully.")
    
    # Load LC model
    lc_model = get_lc_model(config)
    logger.info("Loading LC pretrained weights for sliding window inference...")
    lc_weights_path = config.get('lc_model', {}).get('pretrained_weights', None)
    
    if lc_weights_path is None:
        raise ValueError(
            "LC pretrained weights path must be specified in the config under "
            "'lc_model.pretrained_weights' when running LC sliding window inference."
        )
    
    lc_weights = torch.load(lc_weights_path, map_location=device, weights_only=True)
    lc_model.load_state_dict(lc_weights, strict=True)
    lc_model.eval()
    logger.info("LC pretrained weights loaded successfully.")
    
    # Get paths from config
    input_path = Path(config.get('data', {}).get('input_path'))
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Initialize processor
    logger.info("Initializing LC sliding window processor...")
    processor = LCSlidingWindowProcessor(config, sr_model, lc_model)
    
    # Determine output paths
    output_path = config.get('data', {}).get('output_path', None)
    if output_path is None:
        output_dir = config['paths']['out_path']
        output_path = output_dir / f"{input_path.stem}_lc.tif"
    else:
        output_path = Path(output_path)
    
    save_probs = config.get('inference', {}).get('save_probs', False)
    if save_probs:
        logger.info("LC sliding window inference will save probability maps in addition to predictions.")
        output_probs_path = output_path.with_name(f"{output_path.stem}_preds.tif")
    else:
        output_probs_path = None
    
    # Process file
    logger.info(f"Processing input file: {input_path}")
    logger.info(f"Prediction output will be saved to: {output_path}")
    if output_probs_path:
        logger.info(f"Probability output will be saved to: {output_probs_path}")
    
    processor.process_file(
        input_path=input_path,
        output_pred_path=output_path,
        output_probs_path=output_probs_path
    )
    
    logger.info("Sliding window LC inference complete.")


if __name__ == "__main__":
    main()