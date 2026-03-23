# s2flow — Flow Matching for Sentinel‑2 4× Super‑Resolution (and Land Cover)

**s2flow** is a research codebase for:

1. 4× super‑resolution/enhancement of 4‑band imagery (RGBN)
2. Land cover semantic segmentation

It supports both tile-based workflows and large GeoTIFF sliding-window inference.

## Table of contents

- [s2flow — Flow Matching for Sentinel‑2 4× Super‑Resolution (and Land Cover)](#s2flow--flow-matching-for-sentinel2-4-superresolution-and-land-cover)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [Python version](#python-version)
    - [Install (editable)](#install-editable)
    - [GPU notes](#gpu-notes)
  - [Repository map (what each folder does)](#repository-map-what-each-folder-does)
  - [Config-first workflow (how the CLI runs jobs)](#config-first-workflow-how-the-cli-runs-jobs)
  - [Job types (train/eval/inference) and the code they call](#job-types-trainevalinference-and-the-code-they-call)
    - [Super-resolution (SR)](#super-resolution-sr)
    - [Land cover (LC)](#land-cover-lc)
  - [Inference (deep dive; most important section)](#inference-deep-dive-most-important-section)
    - [SR inference on a directory of tiles (job: sr\_inference)](#sr-inference-on-a-directory-of-tiles-job-sr_inference)
    - [SR inference on a large GeoTIFF via sliding windows (job: sr\_sliding\_window)](#sr-inference-on-a-large-geotiff-via-sliding-windows-job-sr_sliding_window)
    - [LC inference on a directory of tiles (job: lc\_inference)](#lc-inference-on-a-directory-of-tiles-job-lc_inference)
    - [SR→LC inference on a large GeoTIFF via sliding windows (job: lc\_sliding\_window)](#srlc-inference-on-a-large-geotiff-via-sliding-windows-job-lc_sliding_window)
  - [Config reference (every section and key that matters)](#config-reference-every-section-and-key-that-matters)
    - [`job`](#job)
    - [`paths` (auto-created)](#paths-auto-created)
    - [`data`](#data)
    - [`sr_model`](#sr_model)
    - [`lc_model`](#lc_model)
    - [`sampling`](#sampling)
    - [`inference` (sliding-window settings + LC palette)](#inference-sliding-window-settings--lc-palette)
    - [`hyperparameters` (used outside training too)](#hyperparameters-used-outside-training-too)
  - [Data formats and expected parquet schemas](#data-formats-and-expected-parquet-schemas)
    - [Shapefiles vs GeoParquet (`samples.par`)](#shapefiles-vs-geoparquet-samplespar)
      - [Raster label GeoTIFF requirements (LC)](#raster-label-geotiff-requirements-lc)
      - [GeoParquet schema requirements](#geoparquet-schema-requirements)
    - [GeoTIFF expectations](#geotiff-expectations)
    - [SR parquet (used in `sr_train` and `sr_eval`)](#sr-parquet-used-in-sr_train-and-sr_eval)
    - [LC parquet (used in `lc_train` and `lc_eval`)](#lc-parquet-used-in-lc_train-and-lc_eval)
  - [Models and sampling (how SR sampling works)](#models-and-sampling-how-sr-sampling-works)
    - [Flow Matching + ODE solvers (`euler`/`heun`/`midpoint`/`rk4`)](#flow-matching--ode-solvers-eulerheunmidpointrk4)
    - [DDPM/DDIM](#ddpmddim)
    - [GAN mode (RRDBNet)](#gan-mode-rrdbnet)
  - [Training and evaluation](#training-and-evaluation)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [PCA for perceptual metrics (LPIPS/DISTS)](#pca-for-perceptual-metrics-lpipsdists)
  - [SLURM sweeps (batch experiments)](#slurm-sweeps-batch-experiments)
  - [Common gotchas and troubleshooting](#common-gotchas-and-troubleshooting)
    - [“SR inference didn’t make my image 4× bigger”](#sr-inference-didnt-make-my-image-4-bigger)
    - [Sliding-window band order](#sliding-window-band-order)
    - [LC directory inference may crash with AMP enabled](#lc-directory-inference-may-crash-with-amp-enabled)
    - [Sliding-window output path must not already exist](#sliding-window-output-path-must-not-already-exist)
    - [Performance vs quality (SR)](#performance-vs-quality-sr)

## Installation

### Python version

This project requires Python 3.13+ (see `pyproject.toml`).

### Install (editable)

From the repository root:

```bash
pip install -e .
```

This installs a console command:

- `s2flow` → `s2flow.cli:main` (defined in `pyproject.toml`, implemented in `src/s2flow/cli.py`)

### GPU notes

- If CUDA is available, s2flow will use it by default (device selection is in `src/s2flow/utils.py`).
- Mixed precision (AMP) is enabled by default in several inference/training paths when `hyperparameters.use_amp: true`.

## Repository map (what each folder does)

Top-level folders you will interact with most:

- `configs/`: ready-to-run YAML configs for training, evaluation, and inference.
	- Includes inference presets by solver and step count in `configs/inference_steps/`.
- `data/`: example data roots and parquet split files used by configs.
	- Your actual data may live elsewhere; configs must be updated accordingly.
- `runs/` and `logs/`: output and logging directories created automatically per job run.
- `output/`: example output locations used by some configs (not required; you can choose your own).

Core Python package:

- `src/s2flow/`
	- `cli.py`: CLI entry point; loads config; dispatches job types; sets up run directories; copies config to logs.
	- `models.py`: constructs SR and LC models.
	- `engine/`
		- `inference.py`: “simple” directory inference for SR and LC.
		- `sliding_window.py`: sliding-window inference for large rasters (SR only, or SR→LC).
		- `sampling.py`: SR samplers/solvers (Euler/Heun/Midpoint/RK4/DDIM/DDPM + GAN mode).
		- `training.py`: training loops and trainers.
		- `eval.py`: SR and LC evaluation.
	- `data/`
		- `datasets.py`: dataset classes + dataloader builders for training jobs.
		- `transforms.py`: optional spatial augmentation.
		- `utils.py`: scaling helper used across code.
		- `pca.py`: PCA projection layer used for LPIPS/DISTS on multispectral images.
	- `metrics.py` and `loss.py`: metrics and losses.
	- `slurm.py`: sweep utilities for many-run experiments on SLURM clusters.

## Config-first workflow (how the CLI runs jobs)

Everything runs from a YAML config file. You always call:

```bash
s2flow --config path/to/config.yaml
```

What the CLI does (see `src/s2flow/cli.py`):

- Loads YAML into a Python dict.
- Requires `job.name` and `job.type`.
- Creates:
	- `job.log_dir/job.name/`
	- `job.out_dir/job.name/`
- Copies your config into the log folder as `config.yaml` (for provenance).
- Adds a `paths` section to the in-memory config:
	- `paths.log_path`
	- `paths.out_path`
- Dispatches to the job handler based on `job.type`.
- Optionally writes a `COMPLETE` file to the run output directory.

## Job types (train/eval/inference) and the code they call

The allowed job types are enforced in `src/s2flow/cli.py`.

### Super-resolution (SR)

- `sr_train`
	- Model: `get_sr_model` in `src/s2flow/models.py`
	- Trainer chosen by config:
		- If config contains `discriminator_model`: Real-ESRGAN trainer (`RealESRGANTrainer`) in `src/s2flow/engine/training.py`
		- Else if `sampling.solver: ddpm`: `DDPMSRTrainer`
		- Else: `FlowMatchingSRTrainer`
- `sr_eval`
	- Evaluation: `sr_model_evaluation` in `src/s2flow/engine/eval.py`
- `sr_inference`
	- Directory/tile inference: `simple_sr_model_inference` in `src/s2flow/engine/inference.py`
- `sr_sliding_window`
	- Large raster inference: `SRSlidingWindowProcessor` in `src/s2flow/engine/sliding_window.py`

### Land cover (LC)

- `lc_train`
	- Model: `get_lc_model` in `src/s2flow/models.py`
	- Trainer: `LandCoverTrainer` in `src/s2flow/engine/training.py`
- `lc_eval`
	- Evaluation: `lc_model_evaluation` in `src/s2flow/engine/eval.py`
- `lc_inference`
	- Directory/tile inference: `simple_lc_model_inference` in `src/s2flow/engine/inference.py`
- `lc_sliding_window`
	- Large raster SR→LC inference: `LCSlidingWindowProcessor` in `src/s2flow/engine/sliding_window.py`

## Inference (deep dive; most important section)

There are two fundamentally different inference styles:

1. **Simple (directory-based) inference**
	 - Input: many GeoTIFF tiles in a folder
	 - Output: GeoTIFF tiles in an output folder (mirrors subfolders)
	 - Code: `src/s2flow/engine/inference.py`
2. **Sliding-window inference**
	 - Input: one large GeoTIFF (e.g., a full S2 composite tile)
	 - Output: one large GeoTIFF (SR output or LC predictions)
	 - Code: `src/s2flow/engine/sliding_window.py`

Important SR concept: “SR inference” in this repo is sampling-based.
For Flow Matching / DDPM / DDIM modes, SR is not a single forward pass. The sampler iteratively refines a latent/state for `sampling.num_steps` steps using one of the solvers in `src/s2flow/engine/sampling.py`.

### SR inference on a directory of tiles (job: sr_inference)

Use this if you have a directory of tiles you want to run SR on in batches.

**Code path**

- CLI: SR inference dispatch in `src/s2flow/cli.py`
- SR directory inference: `simple_sr_model_inference` in `src/s2flow/engine/inference.py`
- Sampler: `get_sampler` in `src/s2flow/engine/sampling.py`

**What it assumes about the input tiles**

- Pixel values are in **0–10000** (S2-like scaling).
- The images are already at the “model-ready” spatial resolution.
- It does **not** automatically upsample the input; output tiles will have the same height/width as input tiles.

**What it does (exactly)**

- Finds input files: `data.data_root_path.glob(data.glob_pattern)`
- Reads each GeoTIFF: `rasterio` → NumPy array shaped `[C, H, W]`
- Scales `0..10000 → -1..1` using `scale(...)` from `src/s2flow/data/utils.py`
- Batches tiles into a tensor shaped `[B, C, H, W]` and moves to device
- Calls `sampler.sample(input_batch)`
- Scales outputs `-1..1 → 0..10000`
- Writes outputs to `data.out_root_path`, preserving subpaths relative to `data.data_root_path`

**Minimal SR inference config (template)**

```yaml
job:

  name: my_sr_inference
  type: sr_inference
  log_dir: ./logs
  out_dir: ./runs

data:
  data_root_path: /path/to/input_tiles
  glob_pattern: "**/*.tif"
  out_root_path: /path/to/output_tiles

sr_model:
  model_type: unet
  pretrained_weights: /path/to/sr_weights.pt
  sample_size: 256
  in_channels: 8
  out_channels: 4

hyperparameters:
  micro_batch_size: 8
  use_amp: true

sampling:
  solver: euler
  num_steps: 20
  show_pbar: true
  fixed_noise: false
```

**Ready-made examples**

- Euler presets: `configs/inference_steps/euler/`
- DDPM presets: `configs/inference_steps/ddpm/`
- DDIM presets: `configs/inference_steps/ddim/`
- GAN mode (RRDBNet): `configs/s2flow-srinf-real_esrgan.yaml`

Run it:

```bash
s2flow --config configs/inference_steps/euler/s2flow-srinf_10.yaml
```

### SR inference on a large GeoTIFF via sliding windows (job: sr_sliding_window)

Use this when you have one large raster (often too big for GPU memory) and want one large SR output GeoTIFF.

**Code path**

- CLI: `sr_sliding_window_inference` in `src/s2flow/cli.py`
- Processor: `SRSlidingWindowProcessor` in `src/s2flow/engine/sliding_window.py`

**What it assumes about the input GeoTIFF**

- Values are in **0–10000**.
- Band order: the processor’s `process_file` defaults to `correct_band_order=True` and reorders bands from **BGRN → RGBN**.
	- If your file is already RGBN, this reorder would be wrong (currently not exposed as a config switch; it is a function parameter).

**How sliding window SR works (step-by-step, as implemented)**

1. Read entire raster into memory (`rasterio.open(...).read()`).
2. Optional band reorder BGRN → RGBN.
3. Pad the raster by reflection so tile coverage is complete at borders.
4. Enumerate overlapping tiles:
	 - tile size: `inference.tile_size`
	 - stride: `inference.stride`
5. For each tile batch:
	 - Optional TTA:
		 - random horizontal flip
		 - random vertical flip
		 - random 0/90/180/270 rotation
	 - Upsample each tile by `inference.upscale_factor` using bicubic interpolation.
		 - Note: the helper is named `_upsample_lanczos`, but it uses PyTorch bicubic interpolation (see `src/s2flow/engine/sliding_window.py`).
	 - Scale to `-1..1`.
	 - Run sampler for `sampling.num_steps` steps with `sampling.solver`.
	 - Reverse TTA transforms (if applied).
	 - Scale back to `0..10000`.
	 - Multiply each tile by a Gaussian weight mask and accumulate into a global output canvas.
6. Divide accumulated output by accumulated weights (this blends overlaps smoothly).
7. Remove padding.
8. Write output GeoTIFF:
	 - The geotransform pixel size is divided by `upscale_factor`.
	 - Output dtype is written as `int16` (values clipped to `0..10000`).

**How Gaussian blending is configured**

- `inference.gaussian_sigma` is interpreted at output resolution.
- If you do not provide it, the processor defaults to: `tile_size * upscale_factor / 2`.

**Important performance knobs**

- `inference.batch_size`: number of tiles processed at once
- `inference.tile_size` and `inference.stride`: smaller tiles reduce memory; more overlap increases compute
- `sampling.num_steps`: more steps = better quality (often) but slower
- `hyperparameters.use_amp`: mixed precision can speed up inference on modern GPUs

Example config: `configs/s2flow-sr_sliding_window.yaml`

Run it:

```bash
s2flow --config configs/s2flow-sr_sliding_window.yaml
```

**Two important gotchas for sliding window SR**

- Tiles that are entirely zero (or NaN) are skipped by the tile generator.
	- If your data contains legitimate large zero regions, you can unintentionally skip valid tiles.
- If you set `sampling.fixed_noise: true`:
	- the sampler creates a fixed `x_0` tensor hard-coded to shape `(4, 256, 256)` (see `src/s2flow/engine/sampling.py`).
	- This matches the common case: 4 channels, and output tile size = `tile_size * upscale_factor = 64 * 4 = 256`.
	- If you change `tile_size` or `upscale_factor`, fixed-noise may no longer match your tile shape.

### LC inference on a directory of tiles (job: lc_inference)

Use this when you already have tile-sized images and want to create predicted land cover maps.

**Code path**

- CLI: `lc_model_inference` in `src/s2flow/cli.py`
- Directory inference: `simple_lc_model_inference` in `src/s2flow/engine/inference.py`

**What it expects**

- Input tiles as GeoTIFFs, shaped `[C, H, W]`
- The value range depends on `data.source_data`:
	- `s2` or `s2sr`: input scaled from `0..10000 → 0..1`
	- `naip`: input scaled from `0..255 → 0..1`

**What it outputs**

- A 1-band `uint8` prediction map (class IDs).
- Classes are written as **1-indexed** (`prediction + 1`). Internally, models typically use 0-indexed classes.
- Optional color palette:
	- `inference.save_colormap: true`
	- `inference.colormap: {class_id: [R, G, B, A], ...}`

**Minimal LC inference config (template)**

```yaml
job:
  name: my_lc_inference
  type: lc_inference
  log_dir: ./logs
  out_dir: ./runs

data:
  data_root_path: /path/to/input_tiles
  glob_pattern: "**/*.tif"
  out_root_path: /path/to/output_preds
  source_data: s2sr

lc_model:
  model_type: segformer
  in_channels: 4
  num_classes: 7
  pretrained_weights: /path/to/lc_weights.pt

hyperparameters:
  micro_batch_size: 8
  use_amp: false

inference:
  save_colormap: true
  colormap:
    0: [0, 0, 0, 0]
    1: [81, 108, 151, 255]
```

Run it:

```bash
s2flow --config configs/s2flow-simple_lc_inf.yaml
```

**Important known issue**

In `simple_lc_model_inference` the code uses `get_hp_dtype()` but does not import it in `inference.py`. If AMP is enabled for LC inference, this can raise a `NameError`.

Workaround without code changes: set `hyperparameters.use_amp: false` for `lc_inference`.

### SR→LC inference on a large GeoTIFF via sliding windows (job: lc_sliding_window)

Use this when you want land cover predictions for a large Sentinel‑2 raster, but your LC model expects SR-quality inputs. The pipeline is:

S2 raster → tiling → upsample (bicubic) → SR sampling → scale to 0..1 → LC model → stitch probabilities → write predictions

**Code path**

- CLI: `lc_sliding_window_inference` in `src/s2flow/cli.py`
- Processor: `LCSlidingWindowProcessor` in `src/s2flow/engine/sliding_window.py`

**What it outputs**

- Primary output: prediction GeoTIFF (1 band, `uint8`, 1-indexed classes)
- Optional output: per-class probability GeoTIFF (`C` bands, `float32`) when `inference.save_probs: true`

Example config: `configs/s2flow-lc_sliding_window.yaml`

Run it:

```bash
s2flow --config configs/s2flow-lc_sliding_window.yaml
```

**Note about the output-probabilities filename**

The CLI currently constructs a probability path in a slightly confusing way (it uses a `_preds.tif` suffix). If you care about naming, set `inference.save_probs: false` or rename afterward.

## Config reference (every section and key that matters)

This is the “what the code actually reads” map. If a key is not listed here, it is either unused or only used in a niche path.

### `job`

Used by the CLI in `src/s2flow/cli.py`.

- `job.name` (required): name of the run; also the folder name under logs and runs.
- `job.type` (required): selects which pipeline executes.
- `job.log_dir` (optional, default `logs`): base log directory.
- `job.out_dir` (optional, default `runs`): base output directory.
- `job.load_checkpoint` (training jobs only): resume from checkpoint.
- `job.checkpoint_filename` (training jobs only): checkpoint file name, default `checkpoint.pt`.
- `job.cudnn_deterministic` (optional): passed into `torch.backends.cudnn.deterministic`.
- `job.add_completed_file` (optional, default `true`): if true, creates a `COMPLETE` file at the end of the run.

### `paths` (auto-created)

Inserted by the CLI before most jobs run:

- `paths.log_path`: full per-run log directory
- `paths.out_path`: full per-run output directory

Many modules assume these exist (especially training and eval).

### `data`

Varies by job.

SR train/eval:

- `data.samples_par_path`: parquet file describing splits and paths
- `data.data_dir_path`: root folder for paths in parquet
- `data.augmentations`: if set to `spatial`, enables shared flip/rotate transforms during training (see `src/s2flow/data/transforms.py`)
- `data.num_workers`, `data.pin_memory`: dataloader settings

SR inference (directory tiles):

- `data.data_root_path`
- `data.glob_pattern`
- `data.out_root_path`

Sliding window (SR or SR→LC):

- `data.input_path`
- `data.output_path` (optional; if not provided, defaults under `paths.out_path`)

LC train/eval:

- `data.samples_par_path`
- `data.data_dir_path`
- `data.source_data`: `s2`, `naip`, or `s2sr` (controls which input column is used)
- `data.fold`: CV fold ID

LC inference (directory tiles):

- `data.data_root_path`
- `data.glob_pattern`
- `data.out_root_path`
- `data.source_data`: affects scaling (`0..10000` vs `0..255`)

### `sr_model`

Built by `get_sr_model` in `src/s2flow/models.py`.

- `sr_model.model_type`:
	- `unet`: uses Diffusers `UNet2DModel` wrapped by `UNetTensorWrapper`
	- `rrdbnet`: RRDBNet generator used in GAN mode
- `sr_model.pretrained_weights`: required for `sr_eval`, `sr_inference`, `sr_sliding_window`, `lc_sliding_window`
- `sr_model.compile_model` (optional): attempts `torch.compile()` except when hostname is exactly `gcer-a100`

Architecture fields for UNet (common):

- `sample_size`, `in_channels`, `out_channels`, `block_out_channels`, `down_block_types`, `up_block_types`, `layers_per_block`, `norm_num_groups`, `time_embedding_type`

Why `in_channels` is often 8 for UNet:

- In the samplers, SR uses `model_input = concat([x, cond], dim=1)`, where both are 4-channel tensors (so 8 total).

### `lc_model`

Built by `get_lc_model` in `src/s2flow/models.py` via `segmentation-models-pytorch`.

- `lc_model.model_type`: `unet`, `deeplabv3plus`, or `segformer`
- `lc_model.encoder_name` (model-dependent)
- `lc_model.encoder_weights` (often `imagenet`)
- `lc_model.in_channels` (usually 4)
- `lc_model.num_classes`
- `lc_model.pretrained_weights`: required for `lc_eval`, `lc_inference`, `lc_sliding_window`

### `sampling`

Implemented in `src/s2flow/engine/sampling.py`.

- `sampling.gan: true` uses `GANSampler` (RRDBNet path)
- `sampling.solver`: `euler`, `heun`, `midpoint`, `rk4`, `ddim`, `ddpm`
- `sampling.num_steps`: number of steps / scheduler timesteps
- `sampling.fixed_noise`: creates deterministic initial noise for ODE-like samplers (see fixed shape note above)
- `sampling.show_pbar`: progress bar inside sampling loops

### `inference` (sliding-window settings + LC palette)

Used by `src/s2flow/engine/sliding_window.py`.

- `inference.tile_size`: tile size at input resolution (pixels)
- `inference.stride`: tile stride at input resolution
- `inference.batch_size`: tiles per batch
- `inference.upscale_factor`: usually 4
- `inference.gaussian_sigma`: Gaussian blending sigma at output resolution
- `inference.tta`: enable random flip/rotation test-time augmentation
- `inference.tta_passes`: number of passes; outputs are averaged
- `inference.enable_pbar`: show progress bar

LC-specific:

- `inference.save_probs`: save probability raster (`float32`, `C` bands) in addition to predictions
- `inference.colormap`: palette for predictions
- `inference.save_colormap`: whether to write colormap (directory LC inference uses this too)

### `hyperparameters` (used outside training too)

Commonly used for inference:

- `hyperparameters.micro_batch_size`: directory inference batch size
- `hyperparameters.use_amp`: enables AMP contexts in samplers and sliding-window processors

## Data formats and expected parquet schemas

### Shapefiles vs GeoParquet (`samples.par`)

The training code in this repo does **not** read shapefiles directly.

- In configs, `data.samples_par_path` points to a **GeoParquet** file (often named `samples.par`).
- In code, datasets load it with `geopandas.read_parquet(...)` (see `src/s2flow/data/datasets.py`).

If your labels start as a shapefile (polygons/lines/points), the expected pipeline is:

1. **Rasterize your shapefile** into *label rasters* (GeoTIFFs) that align with the imagery tiles you will train on.
2. **Write a GeoParquet/Parquet “samples table”** that lists the (relative) paths to imagery + label rasters and includes split/fold metadata.

#### Raster label GeoTIFF requirements (LC)

For land-cover training (`lc_train`/`lc_eval`), each sample’s `lc_path` must point to a single-band label GeoTIFF where:

- Pixel values are **1-indexed class IDs** (e.g., `1..num_classes`).
  - The loader converts them to 0-indexed internally via `target - 1`.
- The raster grid (CRS, resolution, width/height, transform) should match the corresponding imagery tile.
  - The code assumes per-pixel correspondence; it does not reproject/resample labels at load time.

#### GeoParquet schema requirements

Your GeoParquet may optionally include a `geometry` column (tile footprint polygons) and CRS metadata, but **the loader only uses the path + split/fold columns**.

**SR GeoParquet (used by `sr_train` / `sr_eval`)**

Required columns:

- `split`: must include `train` and `val`
- `input_path`: conditioning image path (relative to `data.data_dir_path`)
- `target_path`: target image path (relative to `data.data_dir_path`)

Optional (recommended):

- `id`: stable identifier used during evaluation outputs

**LC GeoParquet (used by `lc_train` / `lc_eval`)**

Required columns:

- `split`: should include `train`/`val` rows, and may include `test` rows for evaluation
  - Note: the LC dataloader excludes `split == "test"` when creating train/val splits.
- `fold`: integer fold ID used for cross-validation (config uses `data.fold`)
- `lc_path`: label raster path (relative to `data.data_dir_path`)
- One imagery path column matching `data.source_data`:
  - `s2_path` (if `data.source_data: s2`)
  - `naip_path` (if `data.source_data: naip`)
  - `s2sr_path` (if `data.source_data: s2sr`)

Practical note: these path columns can be absolute, but the code always does `data_dir_path / value`, so **relative paths** rooted under `data.data_dir_path` are the simplest and most portable.

### GeoTIFF expectations

- `rasterio` reads as `[C, H, W]`.
- SR paths assume S2-like scaling (`0..10000`) and normalize to `-1..1` for sampling.
- LC paths normalize to `0..1` for segmentation models.
- Scaling helper: `scale(...)` in `src/s2flow/data/utils.py`.

### SR parquet (used in `sr_train` and `sr_eval`)

Dataset: `S2NAIPDataset` in `src/s2flow/data/datasets.py`.

Expected columns:

- `split`: `train` or `val`
- `input_path`: path to conditioning image (relative to `data.data_dir_path`)
- `target_path`: path to target image (relative to `data.data_dir_path`)
- `id`: used as an identifier during evaluation outputs/metrics (recommended)

### LC parquet (used in `lc_train` and `lc_eval`)

Dataset classes in `src/s2flow/data/datasets.py`.

Expected columns:

- `split`: includes `test` for evaluation
- `fold`: integer (for cross-validation splitting)
- `lc_path`: label raster path (labels in files are expected to be 1-indexed; training shifts them to 0-indexed)
- One of:
	- `s2_path` (if `data.source_data: s2`)
	- `naip_path` (if `data.source_data: naip`)
	- `s2sr_path` (if `data.source_data: s2sr`)

## Models and sampling (how SR sampling works)

SR has multiple backends controlled by `sampling`.

### Flow Matching + ODE solvers (`euler`/`heun`/`midpoint`/`rk4`)

Implemented in `src/s2flow/engine/sampling.py`.

Core idea:

- Initialize `x` as noise.
- For each time `t` in `0..1`:
	- concatenate `x` with conditioning image `cond`
	- predict a “velocity” `v = model([x, cond], t)`
	- update `x` according to the solver rule

### DDPM/DDIM

Also in `src/s2flow/engine/sampling.py`, using Diffusers schedulers:

- Predict noise given `[x, cond]` and timestep `t`
- Apply `scheduler.step(...)` to update `x`

### GAN mode (RRDBNet)

If `sampling.gan: true`, sampling uses `GANSampler`:

- It optionally downsamples the conditioning image internally, then calls RRDBNet once.
- This is a feed-forward path (not iterative).

## Training and evaluation

### Training

Training loop classes are in `src/s2flow/engine/training.py`.

Important points for novices:

- Gradient accumulation is used when `hyperparameters.batch_size` is larger than `hyperparameters.micro_batch_size`.
- Checkpoints and models are written under `paths.out_path`.
- Metrics are written to CSV under `paths.log_path`.

### Evaluation

SR evaluation: `src/s2flow/engine/eval.py`

- Runs the sampler on validation samples and computes:
	- L1, PSNR, SSIM, MS-SSIM
	- LPIPS and DISTS computed after a PCA projection of 4-band → 3-band

LC evaluation: `src/s2flow/engine/eval.py`

- Runs on the `test` split
- Computes a confusion matrix and classification metrics (accuracy, precision, recall, F1, mIoU)

### PCA for perceptual metrics (LPIPS/DISTS)

Because LPIPS/DISTS are RGB-based, the repo uses a PCA projection:

- PCA layer and joblib loading: `src/s2flow/data/pca.py`
- Packaged asset: `pca.joblib` (included as package data)
- Metric wrappers: `src/s2flow/metrics.py`

If the PCA joblib is missing, code attempts to fit it from the SR training dataloader (which can take a long time).

## SLURM sweeps (batch experiments)

Sweep helpers live in `src/s2flow/slurm.py`. They are not a CLI job type by default; they are utilities you import to create sweep drivers.

Key classes:

- `SlurmConfig`: SLURM settings (partition, gres, time, max jobs, modules, env activation).
- `BaseJob`: writes per-job configs and an `sbatch` script that runs `s2flow --config <config.yaml>`.
- `BaseSweep`: manages job generation + queue throttling + Ctrl-C handling + optional cancellation.

Configs like `configs/s2flow-sampling_sweep.yaml` are typically used as “base configs” for sweep drivers.

## Common gotchas and troubleshooting

### “SR inference didn’t make my image 4× bigger”

That is expected for `sr_inference`. It assumes your tiles are already at the desired resolution.

Use `sr_sliding_window` for true 4× output size from a coarse GeoTIFF.

### Sliding-window band order

Sliding-window routines reorder BGRN → RGBN by default. If your GeoTIFF is already RGBN, results will look wrong.

### LC directory inference may crash with AMP enabled

Workaround: set `hyperparameters.use_amp: false` for `lc_inference`.

### Sliding-window output path must not already exist

The CLI raises `FileExistsError` if `data.output_path` already exists for sliding-window jobs.

### Performance vs quality (SR)

- Increasing `sampling.num_steps` increases runtime roughly linearly.
- Larger `tile_size` increases memory usage.
- Smaller `stride` increases overlap, improving seam quality but increasing compute.