# GRPO Training for Mathematical Reasoning

This repo contains a simplified GRPO (Group Relative Policy Optimization)
training setup for mathematical reasoning on GSM8K-style data.

## Quick start

1) Install deps:

```
pip install -r requirements.txt
```

2) Download a model:

```
python setup.py
```

This downloads `Qwen2.5-7B-Instruct` into `./models/`. A smaller baseline
(`Qwen2.5-1.5B-Instruct`) is referenced in `models/NOTE.txt` and can also be
used if you have it locally.

3) Run the training script:

```
python grpo_homework.py <MODEL_PATH>
```

## Project layout

- `grpo_homework.py` - Main GRPO training assignment script.
- `gsm8k/` - Local GSM8K dataset files used by the script.
- `models/` - Local model checkpoints (not tracked).
- `saved_model/` - Output checkpoints.
- `notebooks/` - Exploration and analysis notebooks.
- `report/` - Report artifacts.
- `20 epochs/`, `5 epochs/`, `50 epochs/` - Experiment outputs.

## Notes

- Large checkpoints should stay out of git; keep them under `models/`.
