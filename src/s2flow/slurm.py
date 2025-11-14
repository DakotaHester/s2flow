"""
Base classes for parameter sweep functionality.
"""

import os
import subprocess
import time
import socket
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import yaml


class SlurmConfig:
    """SLURM job configuration."""
    
    def __init__(
        self,
        partition: str = "gpu-a100-mig7",
        account: str = "research-abe",
        memory: str = "16G",
        n_tasks: int = 8,
        time: str = "24:00:00",
        gres: str = "gpu:a100_1g.10gb:1",
        mail_user: Optional[str] = None,
        python_env: str = ".venv/bin/activate",
        max_jobs: int = 10,
        modules: Optional[List[str]] = None,
    ):
        self.partition = partition
        self.account = account
        self.memory = memory
        self.n_tasks = n_tasks
        self.time = time
        self.gres = gres
        self.mail_user = mail_user
        self.python_env = python_env
        self.max_jobs = max_jobs
        self.modules = modules or ["cuda", "python/3.10.8"]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SlurmConfig':
        """Create SlurmConfig from dictionary."""
        return cls(**config_dict)


class BaseJob(ABC):
    """Abstract base class for a job in a parameter sweep."""
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        job_params: Dict[str, Any],
        base_log_dir: Path,
        base_out_dir: Path,
        slurm_script_dir: Path,
    ):
        self.base_config = base_config.copy()
        self.job_params = job_params
        self.base_log_dir = base_log_dir
        self.base_out_dir = base_out_dir
        self.slurm_script_dir = slurm_script_dir
        
        # Apply job parameters to config
        self._update_config()
        
        # Generate job metadata
        self.job_name = self._generate_job_name()
        self.job_dir = self._generate_job_dir()
        
        # Setup paths
        self.log_dir = base_log_dir / self.job_dir
        self.out_dir = base_out_dir / self.job_dir
        self.config_path = self.log_dir / "config.yaml"
        self.slurm_script_path = self.slurm_script_dir / f"{self.job_name}.slurm"
        self.log_file = self.log_dir / "slurm.out"
        self.completed_file = self.log_dir / "COMPLETE"
    
    @abstractmethod
    def _update_config(self):
        """Update base config with job-specific parameters."""
        pass
    
    @abstractmethod
    def _generate_job_name(self) -> str:
        """Generate unique job name."""
        pass
    
    @abstractmethod
    def _generate_job_dir(self) -> Path:
        """Generate job directory structure."""
        pass
    
    def _get_command(self) -> List[str]:
        """Get command to execute the job."""
        return ['s2flow', '--config', str(self.config_path)]
    
    def is_completed(self) -> bool:
        """Check if job has already completed."""
        return self.completed_file.exists()
    
    def create_directories(self):
        """Create necessary directories for this job."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.slurm_script_dir.mkdir(parents=True, exist_ok=True)
    
    def write_config(self):
        """Write job-specific config file."""
        job_config = self.base_config.copy()
        job_config['job']['name'] = self.job_name
        job_config['job']['log_dir'] = str(self.log_dir)
        job_config['job']['out_dir'] = str(self.out_dir)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(job_config, f, default_flow_style=False, sort_keys=False)
    
    def create_slurm_script(self, slurm_config: SlurmConfig) -> str:
        """Generate SLURM batch script content."""
        # Module loading
        module_lines = '\n'.join([f"ml {mod}" for mod in slurm_config.modules])
        
        # Mail type configuration
        mail_lines = ""
        if slurm_config.mail_user:
            mail_lines = f"""#SBATCH --mail-type=FAIL
#SBATCH --mail-user={slurm_config.mail_user}"""
        
        # Command to execute
        cmd = ' '.join(self._get_command())
        
        script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n {slurm_config.n_tasks}
#SBATCH --mem={slurm_config.memory}
#SBATCH -p {slurm_config.partition}
#SBATCH -A {slurm_config.account}
#SBATCH -t {slurm_config.time}
#SBATCH --gres={slurm_config.gres}
#SBATCH --job-name={self.job_name}
#SBATCH --output={self.log_file}
{mail_lines}

{module_lines}
source {slurm_config.python_env}
export CUDA_VISIBLE_DEVICES=0

echo "========== SLURM JOB INFO =========="
echo "Job Name: {self.job_name}"
{self._get_info_lines()}
echo "Log dir: {self.log_dir}"
echo "Out dir: {self.out_dir}"
echo "===================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "===================================="

{cmd}

# Mark job as completed
echo "$(date): Job completed successfully" > {self.completed_file}
"""
        return script
    
    def _get_info_lines(self) -> str:
        """Get job-specific info lines for SLURM script header."""
        lines = []
        for key, value in self.job_params.items():
            lines.append(f'echo "{key}: {value}"')
        return '\n'.join(lines)
    
    def submit_slurm(self, slurm_config: SlurmConfig):
        """Submit job to SLURM."""
        self.create_directories()
        self.write_config()
        
        # Write SLURM script
        with open(self.slurm_script_path, 'w') as f:
            f.write(self.create_slurm_script(slurm_config))
        
        # Submit job
        result = subprocess.run(
            ['sbatch', str(self.slurm_script_path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ Submitted {self.job_name} (log: {self.log_file})")
            return True
        else:
            print(f"✗ Failed to submit {self.job_name}: {result.stderr}")
            return False
    
    def run_direct(self):
        """Run job directly (non-SLURM mode)."""
        self.create_directories()
        self.write_config()
        
        print(f"========== RUNNING JOB DIRECTLY ==========")
        print(f"Job Name: {self.job_name}")
        for key, value in self.job_params.items():
            print(f"{key}: {value}")
        print(f"Log dir: {self.log_dir}")
        print(f"Out dir: {self.out_dir}")
        print(f"==========================================")
        
        # Run command
        result = subprocess.run(
            self._get_command(),
            capture_output=True,
            text=True
        )
        
        # Save output to log file
        with open(self.log_file, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)
        
        if result.returncode == 0:
            # Mark as completed
            with open(self.completed_file, 'w') as f:
                f.write(f"{datetime.now()}: Job completed successfully\n")
            print(f"✓ Completed {self.job_name}")
            return True
        else:
            print(f"✗ Failed {self.job_name}")
            return False


class BaseSweep(ABC):
    """Abstract base class for managing parameter sweeps."""
    
    def __init__(
        self,
        base_config_path: str,
        sweep_name: str,
        timestamp: Optional[str] = None,
        slurm_config: Optional[SlurmConfig] = None,
        hostname_check: Optional[str] = None,
    ):
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.sweep_name = sweep_name
        
        # Determine execution mode
        self.hostname = socket.gethostname()
        if hostname_check:
            self.is_slurm = self.hostname == hostname_check
        else:
            # Auto-detect based on common SLURM login node patterns
            self.is_slurm = any(pattern in self.hostname.lower() 
                               for pattern in ['login', 'head', 'submit'])
        
        print(f"Hostname: {self.hostname}")
        print(f"Execution mode: {'SLURM' if self.is_slurm else 'DIRECT'}")
        
        # Setup directories
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.timestamp = timestamp
        self.base_log_dir = Path(f"./logs/{sweep_name}_{timestamp}")
        self.base_out_dir = Path(f"./runs/{sweep_name}_{timestamp}")
        self.slurm_script_dir = Path(f"./slurm_scripts/{sweep_name}_{timestamp}")
        
        # SLURM configuration
        self.slurm_config = slurm_config or SlurmConfig()
        
        # Job tracking
        self.jobs: List[BaseJob] = []
        self.submitted_count = 0
        self.failed_count = 0
    
    @abstractmethod
    def generate_jobs(self):
        """Generate all job configurations for the sweep."""
        pass
    
    def get_queue_size(self) -> int:
        """Get number of jobs currently in SLURM queue."""
        try:
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', os.getlogin())],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Subtract 1 for header line
            return max(0, len(result.stdout.strip().split('\n')) - 1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Warning: Could not query SLURM queue")
            return 0
    
    def wait_for_queue_space(self):
        """Wait until there's space in the queue."""
        while True:
            queue_size = self.get_queue_size()
            if queue_size < self.slurm_config.max_jobs:
                break
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"{queue_size} jobs in queue. Waiting for space...")
            time.sleep(60)
    
    def run(self, skip_completed: bool = True, dry_run: bool = False):
        """Execute the parameter sweep."""
        self.generate_jobs()
        
        if dry_run:
            print(f"\n{'='*50}")
            print(f"DRY RUN: Would submit {len(self.jobs)} jobs")
            print(f"{'='*50}")
            for job in self.jobs:
                status = "(skip: completed)" if job.is_completed() else ""
                print(f"  - {job.job_name} {status}")
            return
        
        for job in self.jobs:
            # Skip if already completed
            if skip_completed and job.is_completed():
                print(f"⊘ Skipping {job.job_name} (already completed)")
                continue
            
            success = False
            if self.is_slurm:
                # Wait for queue space
                self.wait_for_queue_space()
                
                # Submit to SLURM
                success = job.submit_slurm(self.slurm_config)
                time.sleep(1)  # Slight delay to avoid race conditions
            else:
                # Run directly
                success = job.run_direct()
            
            if success:
                self.submitted_count += 1
            else:
                self.failed_count += 1
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print execution summary."""
        mode = "SLURM jobs submitted" if self.is_slurm else "jobs run directly"
        print(f"\n{'='*50}")
        print(f"Sweep: {self.sweep_name}")
        print(f"Total jobs generated: {len(self.jobs)}")
        print(f"Successfully {mode}: {self.submitted_count}")
        if self.failed_count > 0:
            print(f"Failed: {self.failed_count}")
        print(f"Base log directory: {self.base_log_dir}")
        print(f"Base output directory: {self.base_out_dir}")
        print(f"{'='*50}")