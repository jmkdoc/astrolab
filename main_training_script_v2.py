# main_training_script.py

import torch
import lightning.pytorch as pl
import os
import sys # Import sys to potentially exit early

# IMPORTANT: This line ensures your cli.Config(...) call from train_config.py
# is executed, which initializes the singleton Config instance with your settings.
import train_config
import cli # Import cli module itself to access the Config class if needed directly

def main():
    # Now, when you call cli.Config(), it will return the already initialized singleton
    # which holds all the parameters defined in train_config.py.
    main_config = cli.Config()

    print("--- Starting Training Setup ---")
    # Assuming cli.Config has a print_config method, otherwise this line might error
    if hasattr(main_config, 'print_config'):
        main_config.print_config() # Print the loaded configuration for verification
    else:
        print("--- CLI Configuration ---")
        # Fallback if print_config doesn't exist, print key attributes directly
        print(f"  seed_everything: {main_config.seed_everything}")
        print(f"  compile: {main_config.compile}")
        print("  model_factory: <factory function>")
        print("  trainer_factory: <factory function>")
        print("-------------------------")


    # Set seed for reproducibility
    if main_config.seed_everything is not None:
        pl.seed_everything(main_config.seed_everything)
        print(f"Seed set to: {main_config.seed_everything}")

    # 1. Prepare Data Module
    print("Initializing Data Module...")
    datamodule = main_config.trainer_factory().datamodule_factory()

    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # 2. Prepare PyTorch Lightning Module (your model wrapper: lit.CoreLightningTrainer)
    print("Initializing Lightning Model (CoreLightningTrainer)...")
    lit_trainer_module = main_config.trainer_factory() # This instantiates CoreLightningTrainer

    # --- START DEBUG CODE: Check Model Initial Weights ---
    print("\n--- DEBUG: Checking Model Initial Weights ---")
    # Access the underlying model within your LightningModule
    model_to_check = lit_trainer_module.model 
    
    if not isinstance(model_to_check, torch.nn.Module):
        print("WARNING: 'lit_trainer_module.model' is not a torch.nn.Module. Cannot perform detailed weight check.")
        print("  This might happen if your CoreLightningTrainer.model is set up later or is a different type.")
        print("  Proceeding without direct initial weight check. Focus on Step 2 & 3 if issue persists.")
    else:
        found_nan_inf = False
        for name, param in model_to_check.named_parameters():
            if param.requires_grad: # Only check trainable parameters
                if torch.isnan(param).any():
                    print(f"🚨 **CRITICAL ERROR: Initial trainable parameter '{name}' contains NaN values!**")
                    found_nan_inf = True
                if torch.isinf(param).any():
                    print(f"🚨 **CRITICAL ERROR: Initial trainable parameter '{name}' contains Inf values!**")
                    found_nan_inf = True
        
        if found_nan_inf:
            print("\n**Action Required: Model initialized with NaNs/Infs. Debug model/core.py initialization (especially custom layers like RMSNorm, TimeLerp, RWKVFeedForwardSubLayer and their `__init__` methods).**")
            sys.exit(1) # Exit the script if initialization is already bad
        else:
            print("✅ **Initial model parameters appear to be free of NaNs/Infs.**")
    print("--- END DEBUG: Model Initial Weights Check ---\n")
    # --- END DEBUG CODE ---


    # 3. Prepare the PyTorch Lightning Trainer (the pl.Trainer object)
    print("Initializing PyTorch Lightning Trainer...")
    trainer = lit_trainer_module.lightning_trainer_factory()

    # Apply torch.compile if enabled in config
    if main_config.compile:
        try:
            print("Compiling model with torch.compile...")
            lit_trainer_module.model = torch.compile(lit_trainer_module.model, dynamic=True)
            print("Model compilation successful.")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Proceeding without compilation.")

    # 4. Start Training
    print("\n--- Beginning Training ---")
    trainer.fit(lit_trainer_module, datamodule=datamodule)
    print("\n--- Training Completed ---")


if __name__ == "__main__":
    main()