# lit.py
import torch
import lightning.pytorch as pl
from typing import Callable

# Assuming model.core is available
import model.core 

class CoreLightningTrainer(pl.LightningModule):
    def __init__(self, 
                 model_factory: Callable, 
                 optimizer_factory: Callable,
                 lightning_trainer_factory: Callable, # This is actually used by cli.Config to build the Trainer, not passed here.
                 datamodule_factory: Callable): # datamodule_factory also used by cli.Config
        super().__init__()
        self.save_hyperparameters() # Saves model_factory and optimizer_factory to hparams.yaml

        # Initialize the model using the factory
        self.model = model_factory()

        # Store the optimizer factory to create the optimizer in configure_optimizers
        self._optimizer_factory = optimizer_factory
        
        # Loss function (CrossEntropyLoss for language modeling)
        # We ignore padding tokens by setting ignore_index
        # Need to know the pad_token_id from the tokenizer.
        # For GPT2, pad_token_id often needs to be explicitly set.
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) # -100 is default, or use tokenizer.pad_token_id

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        logits = self(input_ids, attention_mask)
        
        # Reshape for CrossEntropyLoss: (N, C, ...) where C is vocab_size
        # logits: (batch_size, sequence_length, vocab_size)
        # labels: (batch_size, sequence_length)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = self._optimizer_factory(self.parameters())
        # You could also configure learning rate schedulers here if needed
        return optimizer

    # This class does not directly use lightning_trainer_factory or datamodule_factory
    # as they are used by the external cli.Config to construct the overall training setup.
    # We keep them in the __init__ signature just to match the expected factory pattern from cli.Config.
