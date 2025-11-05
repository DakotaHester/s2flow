from typing_extensions import Literal
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Tuple, Union, Optional, Dict
import os
from pathlib import Path
import geopandas as gpd
import torch
import rasterio as rio

class S2NAIPDataset(Dataset):

    def __init__(self, 
            samples_par_path: Union[str, Path],
            target_dir_path: Union[str, Path],
            input_dir_path: Union[str, Path],
            transforms: Optional[callable] = None,
            split: Literal['train', 'val'] = 'train',
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
        
        self.samples_par_path = Path(samples_par_path)
        self.target_dir_path = Path(target_dir_path)
        self.input_dir_path = Path(input_dir_path)
        self.transforms = transforms
        
        self.samples_gdf = gpd.read_parquet(self.samples_par_path)
        self.samples_gdf = self.samples_gdf[self.samples_gdf['split'] == split].reset_index(drop=True)
            
    def __len__(self) -> int:
        return len(self.samples_gdf)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sample = self.samples_gdf.iloc[idx]
        target_path = self.target_dir_path / f"{sample['sample_id']:06d}.tif"
        input_path = self.input_dir_path / f"{sample['sample_id']:06d}.tif"
    

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
    
    data_config = config.get("data", None)