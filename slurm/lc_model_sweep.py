from pathlib import Path
from typing import List
from s2flow.slurm import BaseJob, BaseSweep, SlurmConfig


def main() -> None:
    slurm_config = SlurmConfig()
    
    # Initialize the Land Cover Model Sweep
    sweep = LCModelSweep(
        base_config_path='./configs/s2flow-lc_model_sweep.yaml',
        models=['unet', 'deeplabv3plus', 'segformer'],
        data_sources=['s2', 'naip', 's2sr'],
        folds=[0, 1, 2, 3, 4],
        slurm_config=slurm_config,
    )
    
    sweep.run()


class LCModelJob(BaseJob):
    """Job for Land Cover model training with specific architecture, data source, and fold."""

    def _update_config(self):
        """Update config with model_type, source_data, and fold."""
        
        # Ensure sections exist (safety check, though usually present in base yaml)
        if 'lc_model' not in self.base_config:
            self.base_config['lc_model'] = {}
        if 'data' not in self.base_config:
            self.base_config['data'] = {}

        # Update specific parameters
        self.base_config['lc_model']['model_type'] = self.job_params['model']
        self.base_config['data']['source_data'] = self.job_params['source']
        self.base_config['data']['fold'] = self.job_params['fold']

    def _generate_job_name(self) -> str:
        """Generate unique SLURM job name."""
        model = self.job_params['model']
        source = self.job_params['source']
        fold = self.job_params['fold']
        # e.g., s2flow_lc_unet_s2_fold1
        return f"s2flow_lc_{model}_{source}_fold{fold}"

    def _generate_job_dir(self) -> Path:
        """Generate directory structure for logs/runs."""
        model = self.job_params['model']
        source = self.job_params['source']
        fold = self.job_params['fold']
        # e.g., unet/s2/fold_1
        return Path(f"{model}/{source}/fold_{fold}")

    def _get_command(self) -> List[str]:
        """Get s2flow command."""
        return ['s2flow', '--config', str(self.config_path)]


class LCModelSweep(BaseSweep):
    """Parameter sweep for Land Cover model comparison."""

    def __init__(
        self,
        base_config_path: str,
        models: List[str] = None,
        data_sources: List[str] = None,
        folds: List[int] = None,
        timestamp: str = None,
        slurm_config: SlurmConfig = None,
        hostname_check: str = None,
    ):
        super().__init__(
            base_config_path=base_config_path,
            sweep_name="s2flow_lc_model_sweep",
            timestamp=timestamp,
            slurm_config=slurm_config,
            hostname_check=hostname_check,
        )

        # Define parameter space defaults
        self.models = models or ['unet', 'deeplabv3plus', 'segformer']
        self.data_sources = data_sources or ['s2', 'naip', 's2sr']
        self.folds = folds or [0, 1, 2, 3, 4]

    def generate_jobs(self):
        """Generate all LC training jobs."""
        
        # Triple loop over parameters
        for model in self.models:
            for source in self.data_sources:
                for fold in self.folds:
                    
                    job = LCModelJob(
                        base_config=self.base_config,
                        job_params={
                            'model': model,
                            'source': source,
                            'fold': fold,
                        },
                        base_log_dir=self.base_log_dir,
                        base_out_dir=self.base_out_dir,
                        slurm_script_dir=self.slurm_script_dir,
                    )
                    self.jobs.append(job)

        print(f"Generated {len(self.jobs)} Land Cover training jobs")
        print(f"  Models: {', '.join(self.models)}")
        print(f"  Sources: {', '.join(self.data_sources)}")
        print(f"  Folds: {self.folds}")


if __name__ == "__main__":
    main()