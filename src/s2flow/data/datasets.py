from abc import ABC, ABCMeta, abstractmethod
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


class BaseDataset(Dataset, ABC):
    """
    # Structure (SR example):
    # data_dir/
    # samples.par (defines train/val splits and sample IDs)
    # naip/
    #   000000.tif
    #   ...
    # sentinel2/
    #   000000.tif
    #   ...
    """
    def __init__(self,
        samples_gdf: gpd.GeoDataFrame,
        data_dir_path: Union[Path, str],
        transforms: Optional[callable] = None,
    ) -> None:
        
        self.samples_gdf = samples_gdf
        self.data_dir_path = Path(data_dir_path)
        self.transforms = transforms
        
    def __len__(self) -> int:
        return len(self.samples_gdf)
    
    def transform(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transforms is not None:
            input_tensor, target_tensor = self.transforms(input_tensor, target_tensor)
        return input_tensor, target_tensor
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class S2NAIPDataset(BaseDataset):
    """ Sentinel-2/NAIP Image Pairs for Super-Resolution """
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
        
        input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
        
        return input_tensor, target_tensor


class BaseCPBDataset(BaseDataset, ABC):
    """ Base Chesapeake Bay Land Cover Dataset Class """
    
    input_col_name: str  # Must be defined in subclasses

    def __init_subclass__(cls, **kwargs):
        """Ensure subclasses define input_col_name."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'input_col_name') or cls.input_col_name is None:
            raise TypeError(f"{cls.__name__} must define 'input_col_name' class attribute")
    
    def __getitem__(self, idx: int):
        sample = self.samples_gdf.iloc[idx]
        input_path = self.data_dir_path / sample[self.input_col_name]
        target_path = self.data_dir_path / sample['lc_path']
        
        input_image = rio.open(input_path).read()  # [C, H, W]
        target_data = rio.open(target_path).read(1)  # [H, W]
                
        input_tensor = torch.from_numpy(input_image).float()
        target_tensor = torch.from_numpy(target_data).long() - 1 # Shift LC classes to start at 0
        
        input_tensor = scale(
            input_tensor, 
            in_range=(0, 10000) if self.input_col_name in ('s2_path', 's2sr_path') else (0, 255),
            out_range=(0.0, 1.0)
        )
        
        input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
        return input_tensor, target_tensor
    

class S2CPBDataset(BaseCPBDataset):
    """ Sentinel-2/Chesapeake Bay Land Cover Pairs """
    input_col_name = 's2_path'
        

class NAIPCPBDataset(BaseCPBDataset):
    """ NAIP/Chesapeake Bay Land Cover Pairs """
    input_col_name = 'naip_path'


class S2SRCPBDataset(BaseCPBDataset):
    """ Sentinel-2 Super-Resolution/Chesapeake Bay Land Cover Pairs """
    input_col_name = 's2sr_path'


def create_dataloaders(config: Dict[str, Any], train_dataset: Dataset, val_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
    data_config = config.get("data", {})
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
    
    return create_dataloaders(config, train_dataset, val_dataset)


def get_lc_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    
    logger.info("Setting up land cover data loaders...")
    
    data_config = config.get("data", None)
    samples_gdf = gpd.read_parquet(data_config['samples_par_path'])
    cv_samples_gdf = samples_gdf.loc[samples_gdf['split'] != 'test'].reset_index(drop=True)
    
    source_data = data_config.get('source_data', None)  # options: s2, naip, s2sr
    if source_data == 's2':
        DatasetClass = S2CPBDataset
    elif source_data == 'naip':
        DatasetClass = NAIPCPBDataset
    elif source_data == 's2sr':
        DatasetClass = S2SRCPBDataset
    else:
        raise ValueError(f"Unknown source_data: {source_data}. Must be one of 's2', 'naip', 's2sr'.")

    fold = data_config.get('fold', 1)
    train_dataset = DatasetClass(
        cv_samples_gdf.loc[cv_samples_gdf['fold'] != fold].reset_index(drop=True),
        data_config.get('data_dir_path', './data/cpb_lc'),
        transforms=get_transforms(config)
    )
    
    val_dataset = DatasetClass(
        cv_samples_gdf.loc[cv_samples_gdf['fold'] == fold].reset_index(drop=True),
        data_config.get('data_dir_path', './data/cpb_lc'),
        transforms=None
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)} samples.")
    logger.info(f"Validation dataset size: {len(val_dataset)} samples.")
    
    return create_dataloaders(config, train_dataset, val_dataset)
    
    
def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    
    job_type = config.get("job", {}).get("type", None)
    if job_type is None:
        raise ValueError("Job type must be specified in the config under 'job.type'")
    
    if job_type == 'sr_train':
        return get_sr_dataloaders(config)
    elif job_type == 'lc_train':
        return get_lc_dataloaders(config)
    else:
        raise ValueError(
            f"Data loaders are only implemented for 'sr_train' and 'lc_train' job types, but got '{job_type}'."
        )