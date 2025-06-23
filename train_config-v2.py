# /workspaces/astrolab/train_config.py - UPDATED VERSION

import torch
import lightning.pytorch as pl
import transformers
import model.hparams
import cli
import lit
import norm
import model.core
import posemb
import mask
import dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

BATCH_SIZE = 1
VOCAB_SIZE = 50304
TOKENIZER_FACTORY = lambda: transformers.AutoTokenizer.from_pretrained('gpt2')
MAX_SEQUENCE_LENGTH = 256

LOG_PROJECT = 'gptcore'
LOG_NAME = 'TR_AI_JB'

# --- THE CORE CONFIGURATION ---
config_instance = cli.Config(
    seed_everything = 1337,
    compile = True,

    # --- PULLING FACTORIES TO TOP LEVEL ---
    # These factories are needed directly by CoreLightningTrainer's __init__
    # AND now by config_instance directly.
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
                **attn_sublayer_kwargs,
                attention_factory = lambda **attn_kwargs:model.core.TorchAttention(
                    bias_mask_factory=lambda **mask_kwargs: mask.AlibiMask(**mask_kwargs),
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
    ),

    optimizer_factory = lambda params: torch.optim.Adam(
        params=params,
        lr=6e-4,
        betas=(0.9,0.999),
    ),
    
    # You might want to define a separate lightning_trainer_params dict/object
    # if these are always the same. For now, keeping as a lambda.
    lightning_trainer_factory = lambda: pl.Trainer(
        enable_progress_bar=True,
        max_epochs=-1,
        val_check_interval=1024,
        precision = 'bf16-mixed',
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        log_every_n_steps=20,
        logger = CSVLogger(save_dir='./result', name=LOG_NAME),
        callbacks = [
            ModelCheckpoint(
                dirpath='./result',
                filename='{epoch}-{step}-{val_loss:.2f}',
                save_top_k=1,
                monitor='val_loss',
                mode='min',
                save_last=True,
            ),
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
    # --- END OF PULLED FACTORIES ---

    # Now, trainer_factory simply references the factories defined above.
    # Note: If CoreLightningTrainer expects other arguments not covered by these factories,
    # you'd add them here.
    trainer_factory = lambda: lit.CoreLightningTrainer(
        model_factory=config_instance.model_factory, # Referencing top-level config_instance attributes
        optimizer_factory=config_instance.optimizer_factory,
        lightning_trainer_factory=config_instance.lightning_trainer_factory,
        datamodule_factory=config_instance.datamodule_factory,
    ),
)