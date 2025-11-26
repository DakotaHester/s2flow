from pathlib import Path
from typing import List
from s2flow.slurm import BaseJob, BaseSweep, SlurmConfig


def main() -> None:
    slurm_config = SlurmConfig()
    
    # Initialize the Sweep
    sweep = LCStepSweep(
        base_config_path='./configs/s2flow-lc_model_sweep_sampling_steps.yaml',
        models=['segformer', 'deeplabv3plus', 'unet'],
        num_steps=[1, 10, 20, 30, 40, 50],
        folds=[0, 1, 2, 3, 4],
        slurm_config=slurm_config,
    )
    
    sweep.run()


class LCStepSweepJob(BaseJob):
    """Job for LC training varying sampling steps, model architecture, and folds."""

    def _update_config(self):
        """Update config with model, fold, and dynamic data paths based on num_steps."""
        
        model = self.job_params['model']
        steps = self.job_params['num_steps']
        fold = self.job_params['fold']

        # 1. Update Job Name for internal logging
        # The config has "s2flow_lc_model_sweep_<NUM_STEPS>", we make it specific
        if 'job' in self.base_config:
            self.base_config['job']['name'] = f"s2flow_lc_{model}_steps{steps}_fold{fold}"

        # 2. Update Model Params
        if 'lc_model' not in self.base_config:
            self.base_config['lc_model'] = {}
        self.base_config['lc_model']['model_type'] = model

        # 3. Update Data Paths (Dynamic Step Injection)
        if 'data' not in self.base_config:
            self.base_config['data'] = {}
        
        # Construct the specific paths requested:
        # ./data/cpb_lc_var_steps/cpb_lc_<NUM_STEPS>/samples.par
        base_data_root = "./data/cpb_lc_var_steps"
        step_dir = f"cpb_lc_{steps}"
        
        self.base_config['data']['samples_par_path'] = f"{base_data_root}/{step_dir}/samples.par"
        self.base_config['data']['data_dir_path'] = f"{base_data_root}/{step_dir}/"
        
        # Set fold and force source_data to s2sr as per prompt
        self.base_config['data']['fold'] = fold
        self.base_config['data']['source_data'] = 's2sr'

    def _generate_job_name(self) -> str:
        """Generate unique SLURM job name."""
        model = self.job_params['model']
        steps = self.job_params['num_steps']
        fold = self.job_params['fold']
        # e.g., lc_step20_segformer_f0
        return f"lc_step{steps}_{model}_f{fold}"

    def _generate_job_dir(self) -> Path:
        """Generate directory structure for logs/runs."""
        model = self.job_params['model']
        steps = self.job_params['num_steps']
        fold = self.job_params['fold']
        # Structure: steps_X / model / fold_Y
        return Path(f"steps_{steps}/{model}/fold_{fold}")

    def _get_command(self) -> List[str]:
        """Get s2flow command."""
        return ['s2flow', '--config', str(self.config_path)]


class LCStepSweep(BaseSweep):
    """Parameter sweep for Sampling Steps impact on LC Models."""

    def __init__(
        self,
        base_config_path: str,
        models: List[str] = None,
        num_steps: List[int] = None,
        folds: List[int] = None,
        timestamp: str = None,
        slurm_config: SlurmConfig = None,
        hostname_check: str = None,
    ):
        super().__init__(
            base_config_path=base_config_path,
            sweep_name="s2flow_lc_sampling_steps_sweep",
            timestamp=timestamp,
            slurm_config=slurm_config,
            hostname_check=hostname_check,
        )

        # Define parameter space defaults
        self.models = models or ['segformer', 'deeplabv3plus', 'unet']
        self.num_steps = num_steps or [1, 10, 20, 30, 40, 50]
        self.folds = folds or [0, 1, 2, 3, 4]

    def generate_jobs(self):
        """Generate all LC training jobs."""
        
        # Triple loop over parameters
        for steps in self.num_steps:
            for model in self.models:
                for fold in self.folds:
                    
                    job = LCStepSweepJob(
                        base_config=self.base_config,
                        job_params={
                            'num_steps': steps,
                            'model': model,
                            'fold': fold,
                        },
                        base_log_dir=self.base_log_dir,
                        base_out_dir=self.base_out_dir,
                        slurm_script_dir=self.slurm_script_dir,
                    )
                    self.jobs.append(job)

        print(f"Generated {len(self.jobs)} LC Sampling Steps jobs")
        print(f"  Steps: {self.num_steps}")
        print(f"  Models: {', '.join(self.models)}")
        print(f"  Folds: {self.folds}")


if __name__ == "__main__":
    main()