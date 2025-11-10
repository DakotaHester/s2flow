from typing_extensions import Literal
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Tuple, Union, Optional, Dict
from pathlib import Path
import geopandas as gpd
import torch
import rasterio as rio

from .transforms import get_transforms
from .utils import scale

from logging import getLogger
logger = getLogger(__name__)

class S2NAIPDataset(Dataset):

    def __init__(self, 
            samples_gdf: gpd.GeoDataFrame,
            data_dir_path: Union[str, Path],
            transforms: Optional[callable] = None,
    ) -> None:
        """
        # Structure:
        # data_dir/
        # samples.par (defines train/val splits and sample IDs)
        # naip/
        #   000000.tif
        #   ...
        # sentinel2/
        #   000000.tif
        #   ...
        """
        
        self.samples_gdf = samples_gdf
        self.data_dir_path = Path(data_dir_path)
        self.transforms = transforms
            
    def __len__(self) -> int:
        return len(self.samples_gdf)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sample = self.samples_gdf.iloc[idx]
        input_path = self.data_dir_path / sample['input_path']
        target_path = self.data_dir_path / sample['target_path']
        
        input_image = rio.open(input_path).read()  # [C, H, W]
        target_image = rio.open(target_path).read()  # [C, H, W
        
        input_tensor = torch.from_numpy(input_image).float()
        target_tensor = torch.from_numpy(target_image).float()
        
        input_tensor = scale(input_tensor, in_range=(0, 10000), out_range=(-1.0, 1.0))
        target_tensor = scale(target_tensor, in_range=(0, 10000), out_range=(-1.0, 1.0))
        
        if self.transforms is not None:
            input_tensor, target_tensor = self.transforms(input_tensor, target_tensor)
        
        return input_tensor, target_tensor
    

class S2CPBDataset(Dataset):
    
    # Sentinel-2/Chesapeake Bay Land Cover Pairs
    
    def __init__(self):
        raise NotImplementedError("S2CPBDataset is not yet implemented.")


class NAIPCPBDataset(Dataset):
    
    # NAIP/Chesapeake Bay Land Cover Pairs
    
    def __init__(self):
        raise NotImplementedError("NAIPCPBDataset is not yet implemented.")


def get_dataloaders(config: Dict[str, Any]) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    
    job_type = config.get("job", {}).get("type", None)
    if job_type is None:
        raise ValueError("Job type must be specified in the config under 'job.type'")
    
    if job_type in ('sr_train', 'sr_eval', 'sr_inference'):
        # Super-resolution data loaders
        return get_sr_dataloaders(config)
    
    elif job_type in ('lc_train', 'lc_eval', 'lc_inference'):
        raise NotImplementedError("Land cover data loaders are not yet implemented.")
    
    else:
        raise ValueError(
            f"Unknown job type: {job_type}. Must be one of 'sr_train'," + \
            "'sr_eval', 'sr_inference', 'lc_train', 'lc_eval', 'lc_inference'."
        )


def get_sr_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    
    logger.info("Setting up super-resolution data loaders...")
    
    data_config = config.get("data", None)
    samples_gdf = gpd.read_parquet(data_config['samples_par_path'])
    
    train_dataset = S2NAIPDataset(
        samples_gdf.loc[samples_gdf['split'] == 'train'].reset_index(drop=True),
        data_config.get('data_dir_path', './data'), 
        transforms=get_transforms(config)
    )
    
    val_dataset = S2NAIPDataset(
        samples_gdf.loc[samples_gdf['split'] == 'val'].reset_index(drop=True),
        data_config.get('data_dir_path', './data'),
        transforms=None
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)} samples.")
    logger.info(f"Validation dataset size: {len(val_dataset)} samples.")
    
    hyperparams_config = config.get("hyperparameters", {})
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparams_config.get('micro_batch_size', 32),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hyperparams_config.get('micro_batch_size', 32),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        drop_last=False
    )
    
    return train_dataloader, val_dataloader