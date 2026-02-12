from pathlib import Path
from typing import List
from s2flow.slurm import BaseJob, BaseSweep, SlurmConfig


def main():
    config = SlurmConfig(
        memory='32G',
        n_tasks=8,
        time='1:00:00',
    )
    
    sweep = RecompressSweep(
        search_dir='./runs/s2_out_hotspots',  # Adjust to your directory containing GeoTIFFs
        slurm_config=config
    )
    sweep.run()

class RecompressJob(BaseJob):
    """Job to run recompress.py on a single GeoTIFF."""

    def _update_config(self):
        # This sweep doesn't use a YAML config, but the base class 
        # may require this to be implemented. 
        pass

    def _generate_job_name(self) -> str:
        # Sanitize filename for SLURM job name
        return f"recompress_{Path(self.job_params['path']).stem}"

    def _generate_job_dir(self) -> Path:
        # Store logs in a flat directory structure
        return Path(f"recompress_logs/{Path(self.job_params['path']).stem}")

    def _get_command(self) -> List[str]:
        # Direct call to your script
        return ['python', 'slurm/recompress.py', '--path', self.job_params['path']]

class RecompressSweep(BaseSweep):
    """Sweep to recompress all GeoTIFFs in a directory."""

    def __init__(self, search_dir: str, slurm_config: SlurmConfig):
        # Pass a dummy config path if BaseSweep requires it
        super().__init__(
            base_config_path='./configs/dummy.yaml', 
            sweep_name="lzw_recompression_sweep",
            slurm_config=slurm_config
        )
        self.search_dir = Path(search_dir)

    def generate_jobs(self):
        # Recursive glob for all TIFs
        files = list(self.search_dir.rglob("*.tif"))
        
        for f in files:
            self.jobs.append(RecompressJob(
                base_config={}, 
                job_params={'path': str(f)},
                base_log_dir=self.base_log_dir,
                base_out_dir=self.base_out_dir,
                slurm_script_dir=self.slurm_script_dir,
            ))
        print(f"Generated {len(self.jobs)} recompression jobs.")


if __name__ == "__main__":
    main()