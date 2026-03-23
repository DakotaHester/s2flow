# s2flow.slurm: Comprehensive Guide

This folder contains the full SLURM-based experiment management system for the s2flow project. It enables scalable, reproducible, and highly configurable parameter sweeps for training, inference, and evaluation on high-performance computing clusters.

---

## Table of Contents

- [s2flow.slurm: Comprehensive Guide](#s2flowslurm-comprehensive-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Architecture](#architecture)
    - [SlurmConfig](#slurmconfig)
    - [BaseJob](#basejob)
    - [BaseSweep](#basesweep)
  - [Sweep \& Job Types](#sweep--job-types)
    - [Land Cover Model Sweeps](#land-cover-model-sweeps)
    - [Sampling Steps Sweeps](#sampling-steps-sweeps)
    - [Super-Resolution Sampling Sweeps](#super-resolution-sampling-sweeps)
    - [Sliding Window Inference Sweeps](#sliding-window-inference-sweeps)
    - [Recompression Sweeps](#recompression-sweeps)
  - [How to Run Sweeps](#how-to-run-sweeps)
  - [How to Create a New Sweep](#how-to-create-a-new-sweep)
  - [Advanced Usage \& Extensibility](#advanced-usage--extensibility)

---

## Overview

The `s2flow.slurm` system is designed to automate and manage large-scale experiments using SLURM. It provides:

- **Parameter sweeps**: Easily run hundreds or thousands of jobs with different configs.
- **Job templating**: Each job gets its own config, log, and output directory.
- **Resource management**: Handles SLURM queue limits, job submission, and monitoring.
- **Interrupt safety**: Graceful handling of Ctrl-C, with optional job cancellation.
- **Reproducibility**: All configs and logs are saved per job.

---

## Architecture

The core logic is implemented in `src/s2flow/slurm.py` and is based on three main classes:

### SlurmConfig

Defines all SLURM resource parameters (partition, memory, GPUs, time, etc). Can be constructed directly or from a dictionary. Used by all sweeps to control job submission.

**Key fields:**
- `partition`, `account`, `memory`, `n_tasks`, `time`, `gres`, `mail_user`, `python_env`, `max_jobs`, `modules`

### BaseJob

Abstract base class for a single job in a sweep. Handles:
- Config generation (YAML per job)
- Directory structure (logs, outputs, scripts)
- SLURM script creation
- Job submission (via `sbatch`) or direct execution
- Status tracking (COMPLETE file)

**To implement a new job type:**
- Subclass `BaseJob`
- Implement `_update_config`, `_generate_job_name`, `_generate_job_dir`

### BaseSweep

Abstract base class for a parameter sweep. Handles:
- Loading base config
- Generating all jobs (calls `generate_jobs`)
- Directory management (logs, outputs, scripts, timestamped)
- SLURM queue management (waits for space, max jobs)
- Interrupt handling (Ctrl-C prompts for cancellation)
- Submission loop (skips completed jobs, dry-run mode)

**To implement a new sweep:**
- Subclass `BaseSweep`
- Implement `generate_jobs` to create all jobs for the sweep

---

## Sweep & Job Types

This folder contains several ready-to-use sweep scripts, each with a specific purpose. All follow the same pattern: define a sweep class, a job class, and a `main()` entry point.

### Land Cover Model Sweeps

- **File:** `lc_model_sweep.py`
- **Purpose:** Train and compare multiple land cover models (e.g., UNet, DeepLabV3+, SegFormer) across different data sources and cross-validation folds.
- **How it works:**
   - Loops over all combinations of model, data source, and fold.
   - Each job updates the config with the correct model, data, and fold.
   - Output and logs are organized by model/source/fold.

### Sampling Steps Sweeps

- **File:** `lc_steps_sweep.py`
- **Purpose:** Evaluate the effect of different sampling steps on land cover models.
- **How it works:**
   - Loops over model architectures, number of steps, and folds.
   - Each job updates the config with the correct number of steps and model.
   - Output is organized by steps/model/fold.

### Super-Resolution Sampling Sweeps

- **File:** `sampling_sweep.py`
- **Purpose:** Evaluate super-resolution models with different samplers (e.g., DDPM, DDIM) and step counts.
- **How it works:**
   - Loops over solvers and step counts.
   - Each job updates the config with the correct solver and steps.
   - Output is organized by solver/steps.

### Sliding Window Inference Sweeps

- **Files:** `inference_sweep.py`, `sr_inference_sweep.py`
- **Purpose:** Run inference over large datasets, splitting by year and MGRS tile.
- **How it works:**
   - Finds all matching input files (e.g., S2 composites) in a directory tree.
   - Each job processes one file/tile, updating input/output paths in the config.
   - Output is organized by year/MGRS.

### Recompression Sweeps

- **Files:** `recompress.py`, `recompress_sweep.py`
- **Purpose:** Batch recompress GeoTIFFs with LZW compression for storage efficiency.
- **How it works:**
   - Finds all `.tif` files in a directory.
   - Each job runs `recompress.py` on one file.
   - Output logs are organized by file.

---

## How to Run Sweeps

1. **Set up your environment:**
    - Activate your Python environment (e.g., `.venv`)
    - Install dependencies: `uv pip install -r requirements.txt` or as needed
2. **Edit the sweep script:**
    - Set paths, config files, and SLURM parameters as needed
3. **Submit the sweep:**
    - For direct Python scripts:
       ```bash
       python slurm/lc_model_sweep.py
       ```
    - For SLURM batch scripts (rare, most are Python):
       ```bash
       sbatch slurm/inference_sweep.py
       ```
4. **Monitor jobs:**
    - `squeue -u $USER` to see running jobs
    - `sacct` for job history
    - `scancel <jobid>` to cancel
5. **Check logs and outputs:**
    - Logs: `logs/<sweep_name>/<timestamp>/...`
    - Outputs: `runs/<sweep_name>/<timestamp>/...`

---

## How to Create a New Sweep

1. **Copy an existing sweep script** (e.g., `lc_model_sweep.py`) as a template.
2. **Define your parameter space** in the sweep class (`generate_jobs`).
3. **Implement a custom job class** if you need to update configs or outputs differently.
4. **Set up the main function** to instantiate your sweep and call `run()`.
5. **Test locally** (set SLURM config to minimal resources, or run on a small subset).
6. **Submit to SLURM** as above.

**Tips:**
- Use the `dry_run` argument to preview jobs without submitting.
- Use `skip_completed=False` to re-run all jobs.

---

## Advanced Usage & Extensibility

- **Custom job logic:**
   - Subclass `BaseJob` to add new config manipulations, output naming, or command lines.
- **Custom sweep logic:**
   - Subclass `BaseSweep` to add new job generation logic, filtering, or grouping.
- **Interrupt handling:**
   - Press Ctrl-C during submission to pause and optionally cancel jobs.
- **Resource tuning:**
   - Adjust `max_jobs` in `SlurmConfig` to control queue pressure.
- **SLURM script customization:**
   - Edit `create_slurm_script` in your job class for custom modules, envs, or pre/post hooks.

---
