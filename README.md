# CY-PNMGAN

3D GAN training code for porous multiphase material reconstruction.

## Environment

- Python `>=3.9`
- CUDA-enabled PyTorch is recommended for training

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

- Input files are expected to be `.raw` volumes.
- Default volume size is `64x64x64`.
- `uint8` real samples are normalized to `[-1, 1]`.
- The loader auto-handles both binary encodings:
  - `0/1`
  - `0/255`

## Training

Run training with explicit paths:

```bash
python train.py \
  --dataset-path /path/to/raw_data \
  --results-dir ./results \
  --training-steps 500000 \
  --batch-size 100 \
  --num-workers 8
```

Resume from latest checkpoint in `results/checkpoints`:

```bash
python train.py \
  --dataset-path /path/to/raw_data \
  --results-dir ./results \
  --resume
```

Resume from a specific checkpoint:

```bash
python train.py \
  --dataset-path /path/to/raw_data \
  --results-dir ./results \
  --resume \
  --resume-checkpoint-path ./results/checkpoints/ckpt_step_10000.pth
```

## Windows Notes

- On Windows, `DataLoader` multiprocessing can be unstable or slower with high worker counts.
- Recommended starting point:
  - `--num-workers 0` for maximum stability
  - `--num-workers 2` if I/O becomes a bottleneck

## Outputs

All outputs are written under `--results-dir`:

- `losses/training_losses.csv`
- `slices/*.png`
- `checkpoints/*.pth`
- `metrics/check.csv`
