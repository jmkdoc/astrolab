# /workspaces/astrolab/main_training_script.py

import train_config # Imports your cli.Config setup
import cli
# Add any other imports needed at the top of this file if not already present
# e.g., import lightning.pytorch as pl (if pl.Trainer is used directly here)

def main():
    print("Seed set to 1337") # These prints likely come from cli.Config setup
    print("Seed set to: 1337")

    main_config = cli.Config._instance()

    print("Initializing Data Module...")
    # 1. Get the CoreLightningTrainer instance (which is your LightningModule)
    core_lightning_module = main_config.trainer_factory()

    # 2. Get the PyTorch Lightning Trainer instance
    #    This factory is defined inside your CoreLightningTrainer config
    pl_trainer = core_lightning_module.lightning_trainer_factory()

    # 3. Get the DataModule
    datamodule = core_lightning_module.datamodule_factory()

    # 4. Start the training process!
    print("Starting training...")
    pl_trainer.fit(model=core_lightning_module, datamodule=datamodule)

    print("Training complete!")

if __name__ == "__main__":
    main()