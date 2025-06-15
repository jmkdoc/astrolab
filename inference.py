# /workspaces/astrolab/inference_script.py

import torch
import transformers
import model.core
import model.hparams
import norm
import mask
# You might need to adjust imports based on where these modules are located
# For example, if 'mask' is inside 'model/', it would be 'model.mask' etc.

# --- Constants (ensure these match your train_config.py exactly) ---
# It's crucial that these hyperparameters match the ones used during training
VOCAB_SIZE = 50304
MAX_SEQUENCE_LENGTH = 1024
TOKENIZER_FACTORY = lambda: transformers.AutoTokenizer.from_pretrained('gpt2')

# --- Helper Function to Load Your Model ---
def load_model_from_checkpoint(checkpoint_path):
    # Recreate the exact model architecture as defined in train_config.py
    hparams = model.hparams.HParams(
        vocab_size = VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        n_layer=12,
        n_head=12,
        d_model=768,
        n_kv_head_ratio = 1,
        d_qk_ratio = 1,
        d_v_ratio = 1,
        dropout=0,
        feedforward_d_model_ratio=3,
    )

    # Note: These factory definitions must be IDENTICAL to those in train_config.py
    # to correctly load the saved weights.
    model_instance = model.core.Decoder(
        hparams = hparams,
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
    )

    # Load the state dictionary from the checkpoint file
    # Use map_location to ensure it loads correctly regardless of device
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # PyTorch Lightning checkpoints often wrap the model's state_dict
    # under a 'state_dict' key, and sometimes prefix keys with 'model.'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present in the state_dict keys
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint # Assume the checkpoint IS the state_dict

    model_instance.load_state_dict(state_dict)
    model_instance.eval() # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)

    return model_instance

# --- Text Generation Function ---
def generate_text(model, tokenizer, prompt, max_tokens_to_generate=50, temperature=1.0):
    model.eval() # Ensure model is in eval mode again
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Move input_ids to the same device as the model (GPU if available)
    device = next(model.parameters()).device # Get model's current device
    input_ids = input_ids.to(device)

    generated_ids = input_ids
    with torch.no_grad(): # Disable gradient calculations for inference
        for _ in range(max_tokens_to_generate):
            # Get model outputs (logits for the next token)
            # Take only up to MAX_SEQUENCE_LENGTH to avoid exceeding model's capacity
            current_input_ids = generated_ids[:, -MAX_SEQUENCE_LENGTH:] # Take last tokens if sequence grows too long
            outputs = model(current_input_ids)

            # The last token's logits are used for the next prediction
            next_token_logits = outputs[:, -1, :] # Shape: [batch_size, vocab_size]

            if temperature == 0: # Greedy decoding (always pick the most probable token)
                next_token = torch.argmax(next_token_logits, dim=-1)
            else: # Sample with temperature (introduces randomness)
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append the predicted token to the sequence
            generated_ids = torch.cat((generated_ids, next_token), dim=-1)

            # Optional: break if an end-of-sequence token is generated
            # if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            #     break

    # Decode the generated token IDs back to text
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# --- Main Execution for Inference ---
if __name__ == '__main__':
    print("--- Starting Inference Example ---")

    # IMPORTANT: Replace 'path/to/your/best_model.ckpt' with the actual path
    # to one of your saved checkpoint files from the 'lightning_logs/' directory.
    # Example: 'lightning_logs/version_0/checkpoints/epoch=9-step=10239.ckpt'
    checkpoint_path = 'best_model.ckpt' # <<< YOU MUST UPDATE THIS LINE!

    # A simple check to ensure the checkpoint file exists
    import os
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at '{checkpoint_path}'.")
        print("Please update the 'checkpoint_path' variable in 'inference_script.py'")
        print("with the correct path to your trained model checkpoint.")
        print("Skipping inference example.")
    else:
        try:
            # Initialize tokenizer
            tokenizer = TOKENIZER_FACTORY()

            # Load the model from the checkpoint
            print(f"Loading model from: {checkpoint_path}")
            model = load_model_from_checkpoint(checkpoint_path)

            # Move model to GPU if available
            if torch.cuda.is_available():
                model.cuda()
                print("Model moved to GPU.")
            else:
                print("Running inference on CPU (GPU not available).")

            print("Model loaded successfully!")

            # Define a prompt for text generation
            prompt = "The quick brown fox jumps over the lazy dog because "
            print(f"\nPrompt: \"{prompt}\"")

            # Generate text
            generated_text = generate_text(
                model,
                tokenizer,
                prompt,
                max_tokens_to_generate=100, # Generate up to 100 new tokens
                temperature=0.8 # Adjust temperature: 0 for greedy, higher for more creativity
            )

            print(f"\nGenerated Text:\n{generated_text}")

        except Exception as e:
            print(f"An error occurred during inference: {e}")
            print("Double-check that the model architecture definition in inference_script.py")
            print("exactly matches your train_config.py and the checkpoint you are loading.")