from pathlib import Path
from typing import Dict, Any, List
from s2flow.slurm import BaseJob, BaseSweep, SlurmConfig


def main() -> None:
    slurm_config = SlurmConfig()
    sweep = SREvalSweep(
        base_config_path='./configs/s2flow-sampling_sweep.yaml',
        solvers=['euler', 'heun', 'midpoint', 'rk4'],
        num_steps_range=(1, 100, 5),
        slurm_config=slurm_config,
    )
    sweep.run()
        
        

class SREvalJob(BaseJob):
    """Job for super-resolution evaluation with specific sampling parameters."""
    
    def _update_config(self):
        """Update config with solver and num_steps."""
        
        if 'sampling' not in self.base_config:
            self.base_config['sampling'] = {}
        self.base_config['sampling']['solver'] = self.job_params['solver']
        self.base_config['sampling']['num_steps'] = self.job_params['num_steps']
    
    def _generate_job_name(self) -> str:
        """Generate job name from parameters."""
        solver = self.job_params['solver']
        num_steps = self.job_params['num_steps']
        return f"s2flow_eval_{solver}_steps{num_steps}"
    
    def _generate_job_dir(self) -> Path:
        """Generate directory structure."""
        solver = self.job_params['solver']
        num_steps = self.job_params['num_steps']
        return Path(f"{solver}/steps_{num_steps}")
    
    def _get_command(self) -> List[str]:
        """Get s2flow command."""
        return ['s2flow', '--config', str(self.config_path)]


class SREvalSweep(BaseSweep):
    """Parameter sweep for super-resolution evaluation."""
    
    def __init__(
        self,
        base_config_path: str,
        solvers: List[str] = ['euler', 'heun', 'midpoint', 'rk4'],
        num_steps_range: tuple = (1, 100, 5),
        timestamp: str = None,
        slurm_config: SlurmConfig = None,
        hostname_check: str = None,
    ):
        super().__init__(
            base_config_path=base_config_path,
            sweep_name="s2flow_sr_eval_sweep",
            timestamp=timestamp,
            slurm_config=slurm_config,
            hostname_check=hostname_check,
        )
        
        # Define parameter space
        self.solvers = solvers or ['euler', 'heun', 'midpoint', 'rk4']
        
        # Parse num_steps range (start, stop, step)
        start, stop, step = num_steps_range
        self.num_steps_list = [1] if start == 1 else []
        if start <= 5:
            self.num_steps_list += list(range(max(5, start), stop + 1, step))
        else:
            self.num_steps_list = list(range(start, stop + 1, step))
    
    def generate_jobs(self):
        """Generate all SR evaluation jobs."""
        for solver in self.solvers:
            for num_steps in self.num_steps_list:
                job = SREvalJob(
                    base_config=self.base_config,
                    job_params={
                        'solver': solver,
                        'num_steps': num_steps,
                    },
                    base_log_dir=self.base_log_dir,
                    base_out_dir=self.base_out_dir,
                    slurm_script_dir=self.slurm_script_dir,
                )
                self.jobs.append(job)
        
        print(f"Generated {len(self.jobs)} SR evaluation jobs")
        print(f"  Solvers: {', '.join(self.solvers)}")
        print(f"  Num steps: {len(self.num_steps_list)} values")


if __name__ == "__main__":
    main()
