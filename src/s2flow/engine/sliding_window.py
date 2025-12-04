import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio as rio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from logging import getLogger
from contextlib import nullcontext

from ..utils import get_device, get_hp_dtype
from ..engine.sampling import get_sampler
from ..data.utils import scale

logger = getLogger(__name__)


class BaseSlidingWindowProcessor(ABC):
    """Abstract base class for sliding window inference processors.
    
    This class provides common functionality for processing large rasters
    using overlapping tiles with Gaussian weighting and test-time augmentation.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """Initialize the base sliding window processor.
        
        Args:
            config: Configuration dictionary containing inference parameters.
        """
        logger.debug("Initializing BaseSlidingWindowProcessor...")
        
        self.config = config
        self.device = get_device()
        logger.debug(f"Device: {self.device}")
        
        # Inference parameters
        inference_config = config.get('inference', {})
        self.tile_size = inference_config.get('tile_size', 64)
        self.stride = inference_config.get('stride', 32)
        self.batch_size = inference_config.get('batch_size', 8)
        self.tta = inference_config.get('tta', True)
        self.tta_passes = inference_config.get('tta_passes', 4)  # Number of TTA passes to average
        self.upscale_factor = inference_config.get('upscale_factor', 4)
        self.enable_pbar = inference_config.get('enable_pbar', True)
        
        # Gaussian sigma - default based on output tile size
        self.gaussian_sigma = inference_config.get('gaussian_sigma', None)
        if self.gaussian_sigma is None:
            self.gaussian_sigma = self.tile_size * self.upscale_factor / 2
        
        logger.debug(f"Tile size: {self.tile_size}, Stride: {self.stride}, Batch size: {self.batch_size}")
        logger.debug(f"Gaussian sigma: {self.gaussian_sigma}, TTA: {self.tta}, TTA passes: {self.tta_passes}")
        logger.debug(f"Upscale factor: {self.upscale_factor}")
        
        # Output tile size after upsampling
        self.output_tile_size = self.tile_size * self.upscale_factor
        self.output_stride = self.stride * self.upscale_factor
        
        # Create Gaussian weight matrix for output resolution
        self.weights = self._create_gaussian_weights(self.output_tile_size, self.gaussian_sigma)
        self.weights = self.weights.to(self.device)
        logger.debug(f"Gaussian weights created with shape: {self.weights.shape}")
        
        # AMP context
        if config.get('hyperparameters', {}).get('use_amp', True):
            self.autocast_ctx = torch.amp.autocast(device_type=self.device.type, dtype=get_hp_dtype())
            logger.info(f"Using AMP with dtype: {get_hp_dtype()}")
        else:
            self.autocast_ctx = nullcontext()
            logger.info("AMP disabled; using full precision.")
    
    def _create_gaussian_weights(self, size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian weight matrix.
        
        Args:
            size: Size of the weight matrix (square).
            sigma: Standard deviation of the Gaussian.
            
        Returns:
            Gaussian weight matrix of shape (size, size).
        """
        x = np.linspace(-size / 2, size / 2, size)
        y = np.linspace(-size / 2, size / 2, size)
        X, Y = np.meshgrid(x, y)
        weights = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        weights /= weights.max()
        weights = weights.astype(np.float32)
        return torch.from_numpy(weights)
    
    def _pad_raster(self, raster_data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Pad raster with reflect padding.
        
        Args:
            raster_data: Input raster of shape (C, H, W).
            
        Returns:
            Tuple of (padded_raster, (pad_top, pad_bottom, pad_left, pad_right)).
        """
        pad_size = self.tile_size
        C, H, W = raster_data.shape
        
        # Calculate padding to ensure full coverage
        pad_h = (self.stride - (H % self.stride)) % self.stride + pad_size
        pad_w = (self.stride - (W % self.stride)) % self.stride + pad_size
        
        pad_top = pad_size
        pad_bottom = pad_h
        pad_left = pad_size
        pad_right = pad_w
        
        logger.debug(f"Original shape: {raster_data.shape}")
        logger.debug(f"Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        
        # Use numpy reflect padding
        padded = np.pad(
            raster_data,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='reflect'
        )
        
        logger.debug(f"Padded shape: {padded.shape}")
        return padded, (pad_top, pad_bottom, pad_left, pad_right)
    
    def _unpad_output(
        self, 
        output: np.ndarray, 
        padding: Tuple[int, int, int, int],
        original_height: int,
        original_width: int
    ) -> np.ndarray:
        """Remove padding from output, accounting for upscale factor.
        
        Args:
            output: Output array of shape (C, H, W) or (H, W).
            padding: (pad_top, pad_bottom, pad_left, pad_right) at input resolution.
            original_height: Original height at input resolution.
            original_width: Original width at input resolution.
            
        Returns:
            Unpadded output at output resolution.
        """
        pad_top, pad_bottom, pad_left, pad_right = padding
        
        # Scale padding to output resolution
        pad_top_out = pad_top * self.upscale_factor
        pad_left_out = pad_left * self.upscale_factor
        
        # Target output size
        target_height = original_height * self.upscale_factor
        target_width = original_width * self.upscale_factor
        
        if output.ndim == 3:
            return output[:, pad_top_out:pad_top_out + target_height, pad_left_out:pad_left_out + target_width]
        else:
            return output[pad_top_out:pad_top_out + target_height, pad_left_out:pad_left_out + target_width]
    
    def _generate_tile_batches(
        self, 
        raster_data: np.ndarray, 
        nodata_value: float = 0.0
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Generate batches of tiles from the raster.
        
        Args:
            raster_data: Input raster of shape (C, H, W).
            nodata_value: Value indicating no data.
            
        Yields:
            Tuples of (batch_tiles, batch_coords) where batch_tiles is (B, C, H, W)
            and batch_coords is a list of (y, x) coordinates.
        """
        height, width = raster_data.shape[1:]
        batch_tiles = []
        batch_coords = []
        
        for y in range(0, height - self.tile_size + 1, self.stride):
            for x in range(0, width - self.tile_size + 1, self.stride):
                tile = raster_data[:, y:y + self.tile_size, x:x + self.tile_size]
                
                # Skip tiles that are entirely nodata
                if np.all(tile == nodata_value) or np.all(np.isnan(tile)):
                    continue
                
                # Skip incomplete tiles
                if tile.shape[1:] != (self.tile_size, self.tile_size):
                    continue
                
                batch_tiles.append(tile)
                batch_coords.append((y, x))
                
                if len(batch_tiles) == self.batch_size:
                    yield (torch.from_numpy(np.stack(batch_tiles)).float(), batch_coords)
                    batch_tiles = []
                    batch_coords = []
        
        # Yield remaining tiles
        if batch_tiles:
            yield (torch.from_numpy(np.stack(batch_tiles)).float(), batch_coords)
    
    def _apply_tta_augmentation(
        self, 
        tiles: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply test-time augmentation to tiles.
        
        Args:
            tiles: Input tiles of shape (B, C, H, W).
            
        Returns:
            Tuple of (augmented_tiles, is_hflip, is_vflip, rot_angle).
        """
        B = tiles.shape[0]
        is_hflip = torch.randint(0, 2, (B,)).bool()
        is_vflip = torch.randint(0, 2, (B,)).bool()
        rot_angle = torch.randint(0, 4, (B,))
        
        augmented = tiles.clone()
        for i in range(B):
            if is_hflip[i]:
                augmented[i] = torch.flip(augmented[i], [2])
            if is_vflip[i]:
                augmented[i] = torch.flip(augmented[i], [1])
            augmented[i] = torch.rot90(augmented[i], rot_angle[i].item(), [1, 2])
        
        return augmented, is_hflip, is_vflip, rot_angle
    
    def _reverse_tta_augmentation(
        self,
        tiles: torch.Tensor,
        is_hflip: torch.Tensor,
        is_vflip: torch.Tensor,
        rot_angle: torch.Tensor
    ) -> torch.Tensor:
        """Reverse test-time augmentation on tiles.
        
        Args:
            tiles: Augmented tiles of shape (B, C, H, W).
            is_hflip: Horizontal flip flags.
            is_vflip: Vertical flip flags.
            rot_angle: Rotation angles (0-3).
            
        Returns:
            De-augmented tiles.
        """
        B = tiles.shape[0]
        result = tiles.clone()
        
        for i in range(B):
            # Reverse rotation first
            result[i] = torch.rot90(result[i], -rot_angle[i].item(), [1, 2])
            # Then reverse flips
            if is_vflip[i]:
                result[i] = torch.flip(result[i], [1])
            if is_hflip[i]:
                result[i] = torch.flip(result[i], [2])
        
        return result
    
    def _upsample_lanczos(self, tiles: torch.Tensor) -> torch.Tensor:
        """Upsample tiles using Lanczos resampling (approximated with bicubic).
        
        Note: PyTorch doesn't have native Lanczos, so we use bicubic which is similar.
        
        Args:
            tiles: Input tiles of shape (B, C, H, W).
            
        Returns:
            Upsampled tiles of shape (B, C, H*upscale_factor, W*upscale_factor).
        """
        return F.interpolate(
            tiles, 
            scale_factor=self.upscale_factor, 
            mode='bicubic', 
            align_corners=False,
            antialias=True
        )
    
    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Return the number of output channels."""
        pass
    
    @abstractmethod
    def _process_batch(
        self, 
        batch_tiles: torch.Tensor, 
        batch_coords: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Process a batch of tiles through the model(s).
        
        Args:
            batch_tiles: Input tiles of shape (B, C, H, W).
            batch_coords: List of (y, x) coordinates for each tile.
            
        Returns:
            Tuple of (weighted_output, output_coords).
        """
        pass
    
    @abstractmethod
    def _get_output_profile(
        self, 
        input_profile: dict, 
        input_transform: rio.Affine
    ) -> dict:
        """Get the output rasterio profile.
        
        Args:
            input_profile: Input raster profile.
            input_transform: Input raster transform.
            
        Returns:
            Output raster profile.
        """
        pass
    
    @abstractmethod
    def _save_output(
        self, 
        output: np.ndarray, 
        output_path: Path, 
        profile: dict
    ) -> None:
        """Save the output to a file.
        
        Args:
            output: Output array.
            output_path: Path to save the output.
            profile: Rasterio profile for the output.
        """
        pass
    
    def process_raster(self, raster_data: np.ndarray) -> np.ndarray:
        """Process a raster array.
        
        Args:
            raster_data: Input raster of shape (C, H, W) with values in range 0-10000.
            
        Returns:
            Output array at upscaled resolution.
        """
        if raster_data.dtype != np.float32:
            raster_data = raster_data.astype(np.float32)
        
        original_height, original_width = raster_data.shape[1:]
        logger.info(f"Processing raster of shape {raster_data.shape}")
        
        # Pad raster
        padded_raster, padding = self._pad_raster(raster_data)
        padded_height, padded_width = padded_raster.shape[1:]
        
        # Output dimensions
        output_height = padded_height * self.upscale_factor
        output_width = padded_width * self.upscale_factor
        
        # Initialize outputs
        output_accumulator = torch.zeros((self.output_channels, output_height, output_width))
        weight_sum = torch.zeros((output_height, output_width))
        
        # Estimate total batches for progress bar
        num_tiles_y = (padded_height - self.tile_size) // self.stride + 1
        num_tiles_x = (padded_width - self.tile_size) // self.stride + 1
        total_tiles = num_tiles_y * num_tiles_x
        total_batches = (total_tiles + self.batch_size - 1) // self.batch_size
        
        # If TTA is enabled, multiply by number of passes
        if self.tta:
            total_batches *= self.tta_passes
        
        logger.info(f"Estimated {total_tiles} tiles, {total_batches} batches (TTA passes: {self.tta_passes if self.tta else 1})")
        
        # Number of TTA passes
        num_passes = self.tta_passes if self.tta else 1
        
        with tqdm(total=total_batches, desc="Processing tiles", 
                  disable=not self.enable_pbar, unit="batch") as pbar:
            for tta_pass in range(num_passes):
                if self.tta:
                    logger.debug(f"TTA pass {tta_pass + 1}/{num_passes}")
                
                # Process batches
                batch_generator = self._generate_tile_batches(padded_raster)
                
                for batch_tiles, batch_coords in batch_generator:
                    weighted_output, output_coords = self._process_batch(batch_tiles, batch_coords)
                    
                    # Accumulate results
                    for idx, (y, x) in enumerate(output_coords):
                        y_end = y + self.output_tile_size
                        x_end = x + self.output_tile_size
                        
                        output_accumulator[:, y:y_end, x:x_end] += weighted_output[idx]
                        weight_sum[y:y_end, x:x_end] += self.weights.cpu()
                    
                    pbar.update(1)
        
        # Normalize by weight sum
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        output_accumulator = output_accumulator / weight_sum.unsqueeze(0)
        
        # Remove padding
        output = self._unpad_output(output_accumulator.numpy(), padding, original_height, original_width)
        
        logger.info(f"Output shape: {output.shape}")
        
        return output
    
    def process_file(
        self, 
        input_path: Path, 
        output_path: Path
    ) -> np.ndarray:
        """Process a raster file and save the output.
        
        Args:
            input_path: Path to input Sentinel-2 GeoTIFF.
            output_path: Path to save output.
            
        Returns:
            Output array.
        """
        logger.info(f"Reading input file: {input_path}")
        
        with rio.open(input_path) as src:
            raster_data = src.read().astype(np.float32)
            input_profile = src.profile.copy()
            input_transform = src.transform
        
        logger.debug(f"Input raster shape: {raster_data.shape}")
        logger.debug(f"Input CRS: {input_profile.get('crs')}")
        
        # Process raster
        output = self.process_raster(raster_data)
        
        # Get output profile
        output_profile = self._get_output_profile(input_profile, input_transform)
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_output(output, output_path, output_profile)
        
        return output


class SRSlidingWindowProcessor(BaseSlidingWindowProcessor):
    """Sliding window processor for super-resolution only.
    
    Applies super-resolution to Sentinel-2 imagery using flow matching
    with Gaussian weighting and optional test-time augmentation.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        sr_model: nn.Module,
    ):
        """Initialize the SR sliding window processor.
        
        Args:
            config: Configuration dictionary containing inference parameters.
            sr_model: Super-resolution model (flow matching).
        """
        super().__init__(config)
        
        logger.debug("Initializing SRSlidingWindowProcessor...")
        
        # SR model
        self.sr_model = sr_model.to(self.device)
        self.sr_model.eval()
        logger.debug("SR model moved to device and set to eval mode.")
        
        # Get sampler for SR model
        self.sampler = get_sampler(config, sr_model)
        logger.debug("SR sampler initialized.")
        
        # Output channels (same as input for SR)
        self._output_channels = config.get('sr_model', {}).get('out_channels', 4)
        logger.debug(f"Output channels: {self._output_channels}")
    
    @property
    def output_channels(self) -> int:
        """Return the number of output channels."""
        return self._output_channels
    
    @torch.no_grad()
    def _process_batch(
        self, 
        batch_tiles: torch.Tensor, 
        batch_coords: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Process a batch of tiles through the SR model.
        
        Args:
            batch_tiles: Input tiles of shape (B, C, H, W) in raw S2 values (0-10000).
            batch_coords: List of (y, x) coordinates for each tile.
            
        Returns:
            Tuple of (weighted_output, output_coords) where weighted_output is (B, C, H_out, W_out)
            and output_coords is scaled to output resolution.
        """
        logger.debug(f"Processing batch of {batch_tiles.shape[0]} tiles for SR")
        
        # Apply TTA if enabled
        if self.tta:
            batch_tiles_aug, is_hflip, is_vflip, rot_angle = self._apply_tta_augmentation(batch_tiles)
        else:
            batch_tiles_aug = batch_tiles
        
        # Step 1: Upsample 4x using Lanczos-like resampling
        upsampled = self._upsample_lanczos(batch_tiles_aug)
        logger.debug(f"Upsampled shape: {upsampled.shape}")
        
        # Step 2: Scale to (-1, 1) for SR model
        sr_input = scale(upsampled, in_range=(0, 10000), out_range=(-1.0, 1.0))
        sr_input = sr_input.to(self.device)
        
        # Step 3: SR Sampling
        with self.autocast_ctx:
            sr_output = self.sampler.sample(sr_input)
        logger.debug(f"SR output shape: {sr_output.shape}")
        
        # Reverse TTA if applied
        if self.tta:
            sr_output = self._reverse_tta_augmentation(sr_output, is_hflip, is_vflip, rot_angle)
        
        # Scale back to original range for output (0-10000)
        sr_output = scale(sr_output, in_range=(-1.0, 1.0), out_range=(0.0, 10000.0))
        
        # Apply Gaussian weighting
        weighted_output = sr_output.cpu() * self.weights.cpu()
        
        # Scale coordinates to output resolution
        output_coords = [(y * self.upscale_factor, x * self.upscale_factor) for y, x in batch_coords]
        
        return weighted_output, output_coords
    
    def _get_output_profile(
        self, 
        input_profile: dict, 
        input_transform: rio.Affine
    ) -> dict:
        """Get the output rasterio profile for SR output.
        
        Args:
            input_profile: Input raster profile.
            input_transform: Input raster transform.
            
        Returns:
            Output raster profile.
        """
        output_profile = input_profile.copy()
        output_profile.update(
            count=self._output_channels,
            dtype='float32',
            transform=rio.Affine(
                input_transform.a / self.upscale_factor,
                input_transform.b,
                input_transform.c,
                input_transform.d,
                input_transform.e / self.upscale_factor,
                input_transform.f
            )
        )
        return output_profile
    
    def _save_output(
        self, 
        output: np.ndarray, 
        output_path: Path, 
        profile: dict
    ) -> None:
        """Save the SR output to a file.
        
        Args:
            output: SR output array of shape (C, H, W).
            output_path: Path to save the output.
            profile: Rasterio profile for the output.
        """
        # Update profile with output dimensions
        profile.update(
            height=output.shape[1],
            width=output.shape[2]
        )
        
        with rio.open(output_path, 'w', **profile) as dst:
            dst.write(output.astype(np.float32))
        
        logger.info(f"Saved SR output to: {output_path}")


class LCSlidingWindowProcessor(BaseSlidingWindowProcessor):
    """Sliding window processor for super-resolution followed by land cover classification.
    
    Applies super-resolution to Sentinel-2 imagery, then performs land cover
    classification on the enhanced imagery.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        sr_model: nn.Module,
        lc_model: nn.Module,
    ):
        """Initialize the LC sliding window processor.
        
        Args:
            config: Configuration dictionary containing inference parameters.
            sr_model: Super-resolution model (flow matching).
            lc_model: Land cover classification model.
        """
        super().__init__(config)
        
        logger.debug("Initializing LCSlidingWindowProcessor...")
        
        # SR model
        self.sr_model = sr_model.to(self.device)
        self.sr_model.eval()
        logger.debug("SR model moved to device and set to eval mode.")
        
        # LC model
        self.lc_model = lc_model.to(self.device)
        self.lc_model.eval()
        logger.debug("LC model moved to device and set to eval mode.")
        
        # Get sampler for SR model
        self.sampler = get_sampler(config, sr_model)
        logger.debug("SR sampler initialized.")
        
        # Number of classes
        self._num_classes = config.get('lc_model', {}).get('num_classes', 7)
        logger.debug(f"Number of classes: {self._num_classes}")
        
        # Colormap for output
        self.colormap = config.get('inference', {}).get('colormap', None)
    
    @property
    def output_channels(self) -> int:
        """Return the number of output channels (classes)."""
        return self._num_classes
    
    @torch.no_grad()
    def _process_batch(
        self, 
        batch_tiles: torch.Tensor, 
        batch_coords: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Process a batch of tiles through SR and LC models.
        
        Args:
            batch_tiles: Input tiles of shape (B, C, H, W) in raw S2 values (0-10000).
            batch_coords: List of (y, x) coordinates for each tile.
            
        Returns:
            Tuple of (weighted_probs, output_coords) where weighted_probs is (B, num_classes, H_out, W_out)
            and output_coords is scaled to output resolution.
        """
        logger.debug(f"Processing batch of {batch_tiles.shape[0]} tiles for SR+LC")
        
        # Apply TTA if enabled
        if self.tta:
            batch_tiles_aug, is_hflip, is_vflip, rot_angle = self._apply_tta_augmentation(batch_tiles)
        else:
            batch_tiles_aug = batch_tiles
        
        # Step 1: Upsample 4x using Lanczos-like resampling
        upsampled = self._upsample_lanczos(batch_tiles_aug)
        logger.debug(f"Upsampled shape: {upsampled.shape}")
        
        # Step 2: Scale to (-1, 1) for SR model
        sr_input = scale(upsampled, in_range=(0, 10000), out_range=(-1.0, 1.0))
        sr_input = sr_input.to(self.device)
        
        # Step 3: SR Sampling
        with self.autocast_ctx:
            sr_output = self.sampler.sample(sr_input)
        logger.debug(f"SR output shape: {sr_output.shape}")
        
        # Step 4: Scale SR output to (0, 1) for LC model
        lc_input = scale(sr_output, in_range=(-1.0, 1.0), out_range=(0.0, 1.0))
        
        # Step 5: LC inference
        with self.autocast_ctx:
            lc_logits = self.lc_model(lc_input)
        lc_probs = F.softmax(lc_logits, dim=1)
        logger.debug(f"LC probs shape: {lc_probs.shape}")
        
        # Reverse TTA if applied
        if self.tta:
            lc_probs = self._reverse_tta_augmentation(lc_probs, is_hflip, is_vflip, rot_angle)
        
        # Apply Gaussian weighting
        weighted_probs = lc_probs.cpu() * self.weights.cpu()
        
        # Scale coordinates to output resolution
        output_coords = [(y * self.upscale_factor, x * self.upscale_factor) for y, x in batch_coords]
        
        return weighted_probs, output_coords
    
    def _get_output_profile(
        self, 
        input_profile: dict, 
        input_transform: rio.Affine
    ) -> dict:
        """Get the output rasterio profile for LC output.
        
        Args:
            input_profile: Input raster profile.
            input_transform: Input raster transform.
            
        Returns:
            Output raster profile.
        """
        output_profile = input_profile.copy()
        output_profile.update(
            count=1,
            dtype='uint8',
            photometric='palette',
            transform=rio.Affine(
                input_transform.a / self.upscale_factor,
                input_transform.b,
                input_transform.c,
                input_transform.d,
                input_transform.e / self.upscale_factor,
                input_transform.f
            ),
            nodata=0,
            compress='lz77'
        )
        return output_profile
    
    def _save_output(
        self, 
        output: np.ndarray, 
        output_path: Path, 
        profile: dict
    ) -> None:
        """Save the LC output to a file.
        
        Args:
            output: LC probability output array of shape (num_classes, H, W).
            output_path: Path to save the output.
            profile: Rasterio profile for the output.
        """
        # Get class predictions from probabilities
        class_predictions = np.argmax(output, axis=0).astype(np.uint8)
        
        # Update profile with output dimensions
        profile.update(
            height=class_predictions.shape[0],
            width=class_predictions.shape[1],
        )
        
        with rio.open(output_path, 'w', **profile) as dst:
            dst.write(class_predictions + 1, 1)  # Add 1 for 1-indexed classes
            if self.colormap is not None:
                dst.write_colormap(1, self.colormap)
        
        logger.info(f"Saved LC prediction output to: {output_path}")
    
    def process_raster(
        self, 
        raster_data: np.ndarray,
        return_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Process a raster array.
        
        Args:
            raster_data: Input raster of shape (C, H, W) with values in range 0-10000.
            return_probs: If True, return (class_probs, class_predictions), else just predictions.
            
        Returns:
            Class predictions array, or tuple of (probs, predictions) if return_probs=True.
        """
        # Call parent method to get probabilities
        probs = super().process_raster(raster_data)
        
        # Get class predictions
        predictions = np.argmax(probs, axis=0).astype(np.uint8) + 1 # 1-indexed
        
        if return_probs:
            return probs, predictions
        return predictions
    
    def process_file(
        self, 
        input_path: Path, 
        output_pred_path: Path,
        output_probs_path: Optional[Path] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Process a raster file and save the outputs.
        
        Args:
            input_path: Path to input Sentinel-2 GeoTIFF.
            output_pred_path: Path to save prediction output.
            output_probs_path: Optional path to save probability output.
            
        Returns:
            Class predictions array, or tuple of (probs, predictions) if probs_path provided.
        """
        logger.info(f"Reading input file: {input_path}")
        
        with rio.open(input_path) as src:
            raster_data = src.read().astype(np.float32)
            input_profile = src.profile.copy()
            input_transform = src.transform
        
        logger.debug(f"Input raster shape: {raster_data.shape}")
        logger.debug(f"Input CRS: {input_profile.get('crs')}")
        
        # Process raster
        probs = super().process_raster(raster_data)
        predictions = np.argmax(probs, axis=0).astype(np.uint8)
        
        # Get output profile
        output_profile = self._get_output_profile(input_profile, input_transform)
        
        # Save prediction output
        output_pred_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_output(probs, output_pred_path, output_profile)
        
        # Save probability output if requested
        if output_probs_path is not None:
            output_probs_path.parent.mkdir(parents=True, exist_ok=True)
            probs_profile = output_profile.copy()
            probs_profile.update(
                count=self._num_classes,
                dtype='float32',
                photometric=None  # Remove palette photometric for probs
            )
            probs_profile.update(
                height=probs.shape[1],
                width=probs.shape[2]
            )
            with rio.open(output_probs_path, 'w', **probs_profile) as dst:
                dst.write(probs.astype(np.float32))
            logger.info(f"Saved LC probability output to: {output_probs_path}")
            
            return probs, predictions
        
        return predictions


# Factory functions for convenience

def get_sr_processor(config: Dict[str, Any], sr_model: nn.Module) -> SRSlidingWindowProcessor:
    """Create an SR sliding window processor.
    
    Args:
        config: Configuration dictionary.
        sr_model: Super-resolution model.
        
    Returns:
        SRSlidingWindowProcessor instance.
    """
    return SRSlidingWindowProcessor(config, sr_model)


def get_lc_processor(
    config: Dict[str, Any], 
    sr_model: nn.Module, 
    lc_model: nn.Module
) -> LCSlidingWindowProcessor:
    """Create an LC sliding window processor.
    
    Args:
        config: Configuration dictionary.
        sr_model: Super-resolution model.
        lc_model: Land cover classification model.
        
    Returns:
        LCSlidingWindowProcessor instance.
    """
    return LCSlidingWindowProcessor(config, sr_model, lc_model)


@torch.no_grad()
def sliding_window_sr_inference(config: Dict[str, Any], sr_model: nn.Module) -> None:
    """Run sliding window SR inference on a Sentinel-2 tile.
    
    Args:
        config: Configuration dictionary.
        sr_model: Super-resolution model.
    """
    logger.info("Starting sliding window SR inference...")
    
    # Get paths from config
    input_path = Path(config.get('data', {}).get('input_path'))
    output_path = config.get('data', {}).get('output_path', None)
    if output_path is None:
        output_dir = config['paths']['out_path']
        output_path = output_dir / f"{input_path.stem}_sr.tif"
    else:
        output_path = Path(output_path)
    
    # Initialize processor
    processor = SRSlidingWindowProcessor(config, sr_model)
    
    # Process file
    processor.process_file(input_path=input_path, output_path=output_path)
    
    logger.info("Sliding window SR inference complete.")


@torch.no_grad()
def sliding_window_lc_inference(
    config: Dict[str, Any], 
    sr_model: nn.Module, 
    lc_model: nn.Module
) -> None:
    """Run sliding window SR+LC inference on a Sentinel-2 tile.
    
    Args:
        config: Configuration dictionary.
        sr_model: Super-resolution model.
        lc_model: Land cover classification model.
    """
    logger.info("Starting sliding window SR+LC inference...")
    
    # Get paths from config
    # Get paths from config
    input_path = Path(config.get('data', {}).get('input_path'))
    output_pred_path = config.get('data', {}).get('output_path', None)
    if output_path is None:
        output_dir = config['paths']['out_path']
        output_path = output_dir / f"{input_path.stem}_sr.tif"
    else:
        output_path = Path(output_path)
    
    save_probs = config.get('inference', {}).get('save_probs', False)
    if save_probs:
        output_probs_path = output_pred_path.parent / f"{input_path.stem}_probs.tif"
    else:
        output_probs_path = None
    
    # Initialize processor
    processor = LCSlidingWindowProcessor(config, sr_model, lc_model)
    
    
    # Process file
    processor.process_file(
        input_path=input_path,
        output_pred_path=output_pred_path,
        output_probs_path=output_probs_path
    )
    
    logger.info("Sliding window SR+LC inference complete.")