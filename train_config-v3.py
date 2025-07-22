# train_config.py

import torch
import lightning.pytorch as pl
import transformers
import sys

# Import necessary modules directly
import model.hparams
import lit
import norm
import model.core-v3
import posemb # Assuming posemb.py contains your positional embedding logic
import mask # Assuming mask.py contains your AlibiMask logic
import dataset

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# --- Configuration Singleton ---
class Config:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        if not self._initialized:
            self.seed_everything = kwargs.get('seed_everything', 1337)
            self.compile = kwargs.get('compile', False)
            self.learning_rate = kwargs.get('learning_rate', 1e-9)
            self.batch_size = kwargs.get('batch_size', 1)
            self.vocab_size = kwargs.get('vocab_size', 50304)
            self.max_sequence_length = kwargs.get('max_sequence_length', 256)
            self.log_name = kwargs.get('log_name', 'TR_AI_JB')
            self.num_workers = kwargs.get('num_workers', 4) # Added num_workers to config

            # Model hyperparameters instance
            self.hparams = model.hparams.HParams(
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                n_layer=2,  # Example: Reduced layers for initial testing
                n_head=12,
                d_model=768,
                n_kv_head_ratio=1,
                d_qk_ratio=1,
                d_v_ratio=1,
                dropout=0,
                feedforward_d_model_ratio=3,
            )

            # Tokenizer for the DataModule
            self.tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

            self._initialized = True

    def print_config(self):
        print("Loaded Configuration:")
        for attr, value in sorted(self.__dict__.items()):
            if not attr.startswith('_') and not callable(value):
                if attr not in ['hparams', 'tokenizer']: # Avoid printing large objects
                    print(f"  {attr}: {value}")
        print(f"  hparams (d_model): {self.hparams.d_model}")
        print(f"  tokenizer: {type(self.tokenizer).__name__}")

    # --- Factories as methods of Config for cleaner access ---

    def create_model(self):
        # Directly instantiate the model with hparams and other components
        return model.core.Decoder(
            hparams=self.hparams,
            embedding_norm_factory=lambda dim: norm.RMSNorm(dim, weight_scaling=False),
            positional_embedding_factory=lambda: posemb.IdentityPositionalEmbedding(), # Using posemb module
            share_embedding_weights=True,
            layer_factory=lambda d_model, hparams: model.core.TransformerLayer(
                d_model=d_model,
                hparams=hparams,
                self_attention_sublayer_factory = lambda **attn_sublayer_kwargs: model.core.AttentionSubLayer(
                    **attn_sublayer_kwargs,
                    attention_factory = lambda **attn_kwargs: model.core.TorchAttention(
                        bias_mask_factory=lambda **mask_kwargs: mask.AlibiMask(**mask_kwargs), # Using mask module
                        **attn_kwargs
                    ),
                    qkv_norm_factory = lambda dim: norm.RMSNorm(dim, weight_scaling=False),
                    time_mixer_factory = lambda dim: model.core.TimeLerp(dim),
                ),
                feedforward_sublayer_factory = lambda d_model, d_feedforward, dropout: model.core.RWKVFeedForwardSubLayer(
                    d_model=d_model,
                    d_feedforward=d_feedforward,
                    dropout=dropout,
                ),
                residual_op_factory = lambda: model.core.ResidualMixOp(
                    dim=d_model,
                    sublayer_norm_factory = lambda dim: norm.RMSNorm(dim, weight_scaling = False)
                ),
            ),
            final_norm_factory=lambda dim: norm.RMSNorm(dim, weight_scaling=False),
        )

    def create_datamodule(self):
        return dataset.DM(
            dataset_path='pile.py',
            tokenizer_factory=lambda: self.tokenizer, # Use the pre-instantiated tokenizer
            batch_size=self.batch_size,
            sequence_length=self.max_sequence_length,
            num_workers=self.num_workers,
            seed=self.seed_everything,
        )

    def create_trainer_module(self):
        # This returns your CoreLightningTrainer (the pl.LightningModule)
        return lit.CoreLightningTrainer(
            model_factory=self.create_model, # Reference the method to create the model
            optimizer_factory=lambda params: torch.optim.Adam(
                params=params,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
            ),
            # The pl.Trainer instance factory
            lightning_trainer_factory=lambda: pl.Trainer(
                enable_progress_bar=True,
                max_epochs=1, # Reduced for testing, consider more for real training
                val_check_interval=10, # Reduced for faster validation check during debug
                precision='32',
                accumulate_grad_batches=1,
                gradient_clip_val=0.5,
                log_every_n_steps=5, # Log more frequently for debug
                logger=CSVLogger(save_dir='./result', name=self.log_name),
                callbacks=[
                    ModelCheckpoint(
                        dirpath='./result/' + self.log_name + '/checkpoints', # Ensure path is distinct
                        filename='{epoch}-{step}-{val_loss:.2f}',
                        save_top_k=1,
                        monitor='val_loss',
                        mode='min',
                        save_last=True,
                    ),
                ],
            ),
            datamodule_factory=self.create_datamodule, # Reference the method to create the datamodule
        )

# Initialize the singleton Config instance
# This is the global configuration object that main_training_script.py will import
Config = Config(
    seed_everything=1337,
    compile=False, # Set to False for initial debugging (compilation adds complexity)
    learning_rate=1e-9, # Keep current low LR for initial test
    batch_size=1, # Keep current low batch size for initial test
    vocab_size=50304,
    max_sequence_length=256,
    log_name='TR_AI_JB_Reduced', # New log name to differentiate
    num_workers=0, # Set to 0 for initial debugging to avoid multiprocessing issues
)