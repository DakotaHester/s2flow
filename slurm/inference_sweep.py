from pathlib import Path
from typing import List
from s2flow.slurm import BaseJob, BaseSweep, SlurmConfig


def main() -> None:
    slurm_config = SlurmConfig(memory='64G')
    
    # Initialize the Inference Sweep
    sweep = LCSlidingWindowSweep(
        base_config_path='./configs/s2flow-lc_sliding_window.yaml',
        search_dir='./data/s2_composites',
        slurm_config=slurm_config,
    )
    
    sweep.run()


class LCSlidingWindowJob(BaseJob):
    """Job for running sliding window inference on a specific S2 composite."""

    def _update_config(self):
        """Update config with specific input/output paths."""
        
        input_path = Path(self.job_params['input_path'])
        year = self.job_params['year']
        mgrs = self.job_params['mgrs']

        # 1. Update Input Path
        if 'data' not in self.base_config:
            self.base_config['data'] = {}
        self.base_config['data']['input_path'] = str(input_path)

        # 2. Update Output Path
        # Target format: ./runs/s2_out/<YEAR>/<MGRS>.tif
        output_dir = Path(f"./runs/s2_out/{year}")
        output_path = output_dir / f"{mgrs}.tif"
        
        self.base_config['data']['output_path'] = str(output_path)

        # 3. Update Job Name for easy SLURM monitoring
        if 'job' in self.base_config:
            self.base_config['job']['name'] = f"lc_inf_{year}_{mgrs}"

    def _generate_job_name(self) -> str:
        """Generate unique SLURM job name."""
        year = self.job_params['year']
        mgrs = self.job_params['mgrs']
        return f"s2flow_inf_{year}_{mgrs}"

    def _generate_job_dir(self) -> Path:
        """Generate directory structure for logs/runs specific to this tile."""
        year = self.job_params['year']
        mgrs = self.job_params['mgrs']
        # Structure: inference / year / MGRS
        return Path(f"inference/{year}/{mgrs}")

    def _get_command(self) -> List[str]:
        """Get s2flow command."""
        return ['s2flow', '--config', str(self.config_path)]


class LCSlidingWindowSweep(BaseSweep):
    """Sweep to process all matching S2 composites in the data directory."""

    def __init__(
        self,
        base_config_path: str,
        search_dir: str,
        timestamp: str = None,
        slurm_config: SlurmConfig = None,
        hostname_check: str = None,
    ):
        super().__init__(
            base_config_path=base_config_path,
            sweep_name="s2flow_lc_inference_sweep",
            timestamp=timestamp,
            slurm_config=slurm_config,
            hostname_check=hostname_check,
        )
        self.search_dir = Path(search_dir)

    def generate_jobs(self):
        """Find files and generate jobs."""
        
        # Glob pattern based on your structure:
        # ./data/s2_composites/*/annual/*/*/s2_composite_mean.tif
        # Corresponds to: root / year / type / mgrs / date_range / filename
        files = list(self.search_dir.glob("*/annual/*/*/s2_composite_mean.tif"))
        
        if not files:
            print(f"WARNING: No files found in {self.search_dir} matching pattern.")
            return

        for file_path in files:
            # Extract Metadata from path
            # file_path: .../2024/annual/18TWP/20240101_20241231/s2_composite_mean.tif
            # .parent -> 20240101_20241231
            # .parent.parent -> 18TWP (MGRS)
            # .parent.parent.parent -> annual
            # .parent.parent.parent.parent -> 2024 (Year)
            
            mgrs_code = file_path.parent.parent.name
            year = file_path.parent.parent.parent.parent.name
            
            # Optional validation to ensure we grabbed the right folders
            if len(mgrs_code) != 5:
                print(f"Skipping weird path: {file_path} (Detected MGRS: {mgrs_code})")
                continue

            job = LCSlidingWindowJob(
                base_config=self.base_config,
                job_params={
                    'input_path': str(file_path),
                    'year': year,
                    'mgrs': mgrs_code
                },
                base_log_dir=self.base_log_dir,
                base_out_dir=self.base_out_dir,
                slurm_script_dir=self.slurm_script_dir,
            )
            self.jobs.append(job)

        print(f"Generated {len(self.jobs)} Inference Jobs")
        # Print a few examples to verify
        if self.jobs:
            print("First 3 jobs:")
            for j in self.jobs[:3]:
                print(f" - {j._generate_job_name()} -> {j.job_params['input_path']}")


if __name__ == "__main__":
    main()
