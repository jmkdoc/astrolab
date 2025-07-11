# train_config.py
import torch # Needed for torch.nn.Identity
import lightning.pytorch as pl # Needed for pl.Trainer
import transformers
# These imports assume you have model/core.py, model/hparams.py, etc. in place
import model.hparams
import cli
import lit
import norm
import model.core
import posemb
import mask # For AlibiMask
import dataset
from lightning.pytorch.callbacks import ModelCheckpoint 
from lightning.pytorch.loggers import CSVLogger # <<< ADD THIS IMPORT
# --- Constants for your model and training ---
BATCH_SIZE = 1
VOCAB_SIZE = 50304 # Based on gpt2 tokenizer vocab size, + special tokens if added
TOKENIZER_FACTORY = lambda: transformers.AutoTokenizer.from_pretrained('gpt2')
MAX_SEQUENCE_LENGTH = 256

LOG_PROJECT = 'gptcore'
LOG_NAME = 'TR_AI_JB'

# --- THE CORE CONFIGURATION ---
# This call initializes the cli.Config singleton with all your model and training parameters.
# It MUST be executed once when your program starts (which happens when main_training_script.py imports this file).
class Config(cli.Config): # Or just 'Config' if it's directly cli.Config
    # ... other config parameters ...

    model_factory = lambda: model.core.Decoder(...)
    optimizer_factory = lambda trainer_module: torch.optim.AdamW(...)
    lightning_trainer_factory = lambda: pl.Trainer(...)
    datamodule_factory = lambda: data.MyDataModule(...)


cli.Config(
    seed_everything = 1337,
    compile = True,

    # Model definition factory
    model_factory = lambda: model.core.Decoder(
        hparams = model.hparams.HParams(
            vocab_size = VOCAB_SIZE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            n_layer=2,
            n_head=12,
            d_model=768,
            n_kv_head_ratio = 1,
            d_qk_ratio = 1,
            d_v_ratio = 1,
            dropout=0,
            feedforward_d_model_ratio=3,
        ),
        embedding_norm_factory=lambda dim: norm.RMSNorm(dim, weight_scaling=False),
        positional_embedding_factory=lambda:torch.nn.Identity(),
        share_embedding_weights=True,
        # The layer_factory needs to pass hparams and d_model to TransformerLayer
        layer_factory=lambda d_model, hparams: model.core.TransformerLayer(
            d_model=d_model,
            hparams=hparams,
            self_attention_sublayer_factory = lambda **attn_sublayer_kwargs: model.core.AttentionSubLayer(
                # Pass arguments like d_model, n_head, d_qk, d_v from TransformerLayer
                **attn_sublayer_kwargs,
                # Explicitly define and pass the factories required by AttentionSubLayer
                attention_factory = lambda **attn_kwargs:model.core.TorchAttention(
                    bias_mask_factory=lambda **mask_kwargs: mask.AlibiMask(**mask_kwargs), # <<< CHANGE THIS LINE
                **attn_kwargs # TorchAttention expects these from AttentionSubLayer
            ),
                qkv_norm_factory = lambda dim: norm.RMSNorm(dim, weight_scaling=False),
                time_mixer_factory = lambda dim: model.core.TimeLerp(dim),
            ),
            # Corrected: RWKVFeedForwardSublayer (capital 'L')
            feedforward_sublayer_factory = lambda d_model, d_feedforward, dropout: model.core.RWKVFeedForwardSubLayer(
                d_model=d_model,
                d_feedforward=d_feedforward,
                dropout=dropout,
            ),
            # Corrected: residual_op_factory passes d_model to ResidualMixOp
            residual_op_factory = lambda: model.core.ResidualMixOp(
                dim=d_model, # Use d_model from the outer layer_factory lambda
                sublayer_norm_factory = lambda dim: norm.RMSNorm(dim, weight_scaling = False)
            ), # <<< ADDED COMMA HERE
        ),
        final_norm_factory=lambda dim: norm.RMSNorm(dim, weight_scaling=False),
    ),

    # Trainer configuration factory
    trainer_factory = lambda: lit.CoreLightningTrainer(
        # The model_factory here is the one used by CoreLightningTrainer.
        # It's a duplicate of the one above, but that's how your original config was structured.
        model_factory = lambda: model.core.Decoder(
            hparams = model.hparams.HParams(
                vocab_size = VOCAB_SIZE,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                n_layer=2,
                n_head=12,
                d_model=768,
                n_kv_head_ratio = 1,
                d_qk_ratio = 1,
                d_v_ratio = 1,
                dropout=0,
                feedforward_d_model_ratio=3,
            ),
            embedding_norm_factory=lambda dim: norm.RMSNorm(dim, weight_scaling=False),
            positional_embedding_factory=lambda:torch.nn.Identity(),
            share_embedding_weights=True,
            layer_factory=lambda d_model, hparams: model.core.TransformerLayer(
                d_model=d_model,
                hparams=hparams,
                self_attention_sublayer_factory = lambda **attn_sublayer_kwargs: model.core.AttentionSubLayer(
                    # Pass arguments like d_model, n_head, d_qk, d_v from TransformerLayer
                    **attn_sublayer_kwargs,
                    # Explicitly define and pass the factories required by AttentionSubLayer
                     attention_factory = lambda **attn_kwargs:model.core.TorchAttention(
                        bias_mask_factory=lambda **mask_kwargs: mask.AlibiMask(**mask_kwargs), # <<< CHANGE THIS LINE
                        **attn_kwargs
            ),
                    qkv_norm_factory = lambda dim: norm.RMSNorm(dim, weight_scaling=False),
                    time_mixer_factory = lambda dim: model.core.TimeLerp(dim),
                ),
                # Corrected: RWKVFeedForwardSublayer (capital 'L')
                feedforward_sublayer_factory = lambda d_model, d_feedforward, dropout: model.core.RWKVFeedForwardSubLayer(
                    d_model=d_model,
                    d_feedforward=d_feedforward,
                    dropout=dropout,
                ),
                # Corrected: residual_op_factory passes d_model to ResidualMixOp
                residual_op_factory = lambda: model.core.ResidualMixOp(
                    dim=d_model, # Use d_model from the outer layer_factory lambda
                    sublayer_norm_factory = lambda dim: norm.RMSNorm(dim, weight_scaling = False)
                ), # <<< ADDED COMMA HERE
            ),
            final_norm_factory=lambda dim: norm.RMSNorm(dim, weight_scaling=False),
        ),
        optimizer_factory = lambda params: torch.optim.Adam(
            params=params,
            lr=1e-5,  # Start with 1e-5 (0.00001) or even 5e-6 (0.000005)
            betas=(0.9,0.999),
        ),
        lightning_trainer_factory = lambda: pl.Trainer(
            enable_progress_bar=True,
            max_epochs=-1,
            val_check_interval=1024,
            precision = 'bf16-mixed',
            accumulate_grad_batches=1,
            gradient_clip_val=0.5,
            log_every_n_steps=20,
            #logger = [], # No loggers enabled by default in your config
            logger = CSVLogger(save_dir='./result', name=LOG_NAME), # <<< CHANGE THIS LINE
            callbacks = [ # <<< MODIFY THIS LIST
                ModelCheckpoint(
                    dirpath='./result', # <<< YOUR DESIRED PATH HERE
                    filename='{epoch}-{step}-{val_loss:.2f}', # Name format for checkpoints
                    save_top_k=1, # Saves only the best K models based on monitor
                    monitor='val_loss', # Metric to monitor (e.g., validation loss)
                    mode='min', # 'min' for loss (lower is better), 'max' for accuracy (higher is better)
                    save_last=True, # Also saves the latest checkpoint as 'last.ckpt'
                ),
                # Add other callbacks here if you have any
            ],
        ),
        datamodule_factory=lambda: dataset.DM(
            dataset_path='pile.py',
            tokenizer_factory=TOKENIZER_FACTORY,
            batch_size=BATCH_SIZE,
            sequence_length=MAX_SEQUENCE_LENGTH,
            num_workers=4,
            seed=32,
        ),
    ),
)
