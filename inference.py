# /workspaces/astrolab/inference-v2.py

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from typing import Callable # Ensure this is imported if you use type hints like Callable

# --- CORRECTED IMPORT PATH FOR CORE_LIGHTNING_TRAINER ---
# Assuming 'lit.py' is in the same directory as 'inference-v2.py',
# or otherwise directly importable from your project root.
from lit import CoreLightningTrainer # <<< Changed from 'model.trainer' to 'lit'

# --- Import your configuration ---
# Make sure 'train_config.py' is also importable from 'inference-v2.py'.
# If it's in a specific subfolder (e.g., 'configs/train_config.py'),
# your import might need to be 'from configs import train_config'.
import train_config 


def load_model_for_inference(checkpoint_path: str, tokenizer_name: str):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pass the factories from your configuration to load_from_checkpoint
    # These factories (model_factory, optimizer_factory, etc.) are defined
    # in your 'train_config.py' file.
    model = CoreLightningTrainer.load_from_checkpoint(
        checkpoint_path,
        # These keyword arguments match the required arguments in CoreLightningTrainer.__init__
        model_factory=train_config.model_factory,
        optimizer_factory=train_config.optimizer_factory,
        lightning_trainer_factory=train_config.lightning_trainer_factory,
        datamodule_factory=train_config.datamodule_factory,
        
        # IMPORTANT: If your CoreLightningTrainer's __init__ also directly takes
        # 'tokenizer' or 'vocab_size' as arguments (and they weren't saved as hparams),
        # you MUST pass them here as well. For instance:
        # tokenizer=tokenizer,
        # vocab_size=len(tokenizer),
    )

    # Set the model to evaluation mode
    model.eval()

    # Move model to appropriate device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        model.to("cuda")
        print("Model moved to GPU.")
    else:
        model.to("cpu")
        print("Model moved to CPU.")

    return model, tokenizer

# --- Rest of your generate_text and main_inference functions ---
# (These parts should remain the same as before)
# ...