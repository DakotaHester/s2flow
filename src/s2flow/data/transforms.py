from typing import Dict, Any, Optional
import torch
import torchvision.transforms.functional as F
from logging import getLogger

logger = getLogger(__name__)

class SpatialDataAugmentations:
    '''
    Simple data augmentation that applies random rotation, horizontal, and vertical flips.
    Applies the same random transforms to all input tensors.
    '''
    @staticmethod
    def __call__(*images: torch.Tensor):
        # Generate random transforms
        do_hflip = torch.rand(1) > 0.5
        do_vflip = torch.rand(1) > 0.5
        rot_angle = torch.randint(0, 4, (1,)).item()

        transformed = []
        for img in images:
            if do_hflip:
                img = F.hflip(img)
            if do_vflip:
                img = F.vflip(img)
            img = torch.rot90(img, rot_angle, dims=[-2, -1])
            transformed.append(img)
        return tuple(transformed)


def get_transforms(config: Dict[str, Any]) -> Optional[callable]:
    logger.debug("Setting up data transforms...")
    augmentations = config.get('data', {}).get('augmentations', {})
    if augmentations == 'spatial':
        logger.info("Using spatial data augmentations.")
        return SpatialDataAugmentations()
    else:
        logger.info("No data augmentations will be applied.")
        return None