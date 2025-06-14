# main_training_script.py

import torch
import lightning.pytorch as pl
import os

# IMPORTANT: This line ensures your cli.Config(...) call from train_config.py
# is executed, which initializes the singleton Config instance with your settings.
import train_config
import cli # Import cli module itself to access the Config class if needed directly

def main():
    # Now, when you call cli.Config(), it will return the already initialized singleton
    # which holds all the parameters defined in train_config.py.
    main_config = cli.Config()

    print("--- Starting Training Setup ---")
    main_config.print_config() # Print the loaded configuration for verification

    # Set seed for reproducibility
    if main_config.seed_everything is not None: # This attribute should now exist!
        pl.seed_everything(main_config.seed_everything)
        print(f"Seed set to: {main_config.seed_everything}")

    # 1. Prepare Data Module
    print("Initializing Data Module...")
    # datamodule_factory is a lambda that returns a dataset.DM instance
    datamodule = main_config.trainer_factory().datamodule_factory()

    # These calls are crucial for data loading setup in LightningDataModule
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # 2. Prepare PyTorch Lightning Module (your model wrapper: lit.CoreLightningTrainer)
    print("Initializing Lightning Model (CoreLightningTrainer)...")
    # main_config.trainer_factory() itself returns an instance of lit.CoreLightningTrainer
    lit_trainer_module = main_config.trainer_factory()

    # 3. Prepare the PyTorch Lightning Trainer (the pl.Trainer object)
    print("Initializing PyTorch Lightning Trainer...")
    trainer = main_config.trainer_factory().lightning_trainer_factory()

    # Apply torch.compile if enabled in config
    if main_config.compile:
        try:
            print("Compiling model with torch.compile...")
            # Compile the LightningModule. This should wrap the internal model.
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