# /workspaces/astrolab/inference-v2.py - UPDATED VERSION

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from typing import Callable

from lit import CoreLightningTrainer
from train_config2 import config_instance


def load_model_for_inference(checkpoint_path: str, tokenizer_name: str = "gpt2"):
    """
    Loads the trained model from a .ckpt file for inference,
    handling state_dict key mismatches due to torch.compile.
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Manual checkpoint loading and state_dict modification ---
    # 1. Load the raw checkpoint file
    # Use map_location='cpu' to avoid GPU memory issues during loading if not strictly needed
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Extract the state_dict and process its keys
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        # Check if the key has the '_orig_mod' prefix and remove it
        if key.startswith("model._orig_mod."):
            # Replace only the first occurrence to avoid issues if 'model._orig_mod.' appears later in the key
            new_key = key.replace("model._orig_mod.", "model.", 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value # Keep keys as they are if no '_orig_mod'

    # 3. Instantiate CoreLightningTrainer manually (without load_from_checkpoint)
    # Ensure all required factories are passed from config_instance
    model = CoreLightningTrainer(
        model_factory=config_instance.model_factory,
        optimizer_factory=config_instance.optimizer_factory,
        lightning_trainer_factory=config_instance.lightning_trainer_factory,
        datamodule_factory=config_instance.datamodule_factory,
        # Add any other direct arguments your CoreLightningTrainer's __init__ might need
        # e.g., tokenizer=tokenizer, vocab_size=len(tokenizer), if they are part of its __init__ signature
    )

    # 4. Load the modified state_dict into the instantiated model
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Model state_dict loaded successfully with strict=True.")
    except RuntimeError as e:
        # Fallback to strict=False if there are other minor mismatches, but this should be rare now
        print(f"Failed to load state_dict with strict=True: {e}. Attempting with strict=False.")
        model.load_state_dict(new_state_dict, strict=False)
        print("Model state_dict loaded successfully with strict=False (warnings may be present).")


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


@torch.no_grad()
def generate_text(model: pl.LightningModule, tokenizer: AutoTokenizer,
                  prompt: str, max_new_tokens: int = 100, temperature: float = 0.8,
                  top_k: int = 50, top_p: float = 0.95):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        # Ensure 'model.model' points to the actual Decoder instance
        logits = model.model(generated_ids)

        next_token_logits = logits[:, -1, :]

        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1)
        else:
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                if sorted_indices_to_remove.shape[1] > 1:
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                next_token_logits = next_token_logits.scatter(
                    -1, sorted_indices, next_token_logits.new_zeros(next_token_logits.shape).fill_(-float("Inf"))
                )
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def main_inference():
    CHECKPOINT_PATH = "./result/last.ckpt"
    TOKENIZER_NAME = "gpt2"

    model, tokenizer = load_model_for_inference(CHECKPOINT_PATH, TOKENIZER_NAME)
    print("Model and tokenizer loaded successfully for inference.")

    print("\nEnter prompts to generate text (type 'exit' to quit):")
    while True:
        user_prompt = input("Prompt: ")
        if user_prompt.lower() == 'exit':
            break

        print("Generating...")
        generated_text = generate_text(model, tokenizer, user_prompt, max_new_tokens=150, temperature=0.7)
        print("\n--- Generated Text ---")
        print(generated_text)
        print("----------------------\n")

if __name__ == "__main__":
    main_inference()