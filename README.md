## RLFT with PPO Reft style

This work is based on these two papers [PPO](https://arxiv.org/abs/1707.06347) and [Reft](https://arxiv.org/abs/2401.08967). This code uses Nutribench dataset for training.

## Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Training PPO

Before training with PPO, you need to edit the `ppo.sh` script to set the correct paths:

```bash
# Edit ppo.sh to set:
# --model_name_or_path: path to your warmed up model
# --output_dir: where you want to save the trained model

# Run the PPO training
sh ppo.sh
```

**Note:** It is recommended to keep all hyperparameters as they are in the script for achieving the good MAE on Nutribench.

## Remove Value Head

After PPO training, you'll need to remove the value head from the model before using it for inference:

```bash
# Edit vinf.py to set:
# src: path to your PPO trained model
# dst: path where to save the model without value head

# Run the script to remove value head
python vinf.py
```

