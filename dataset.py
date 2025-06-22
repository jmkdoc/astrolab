# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import transformers

class DummyTextDataset(Dataset):
    def __init__(self, tokenizer, sequence_length: int, num_samples: int = 1000):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        # Create a simple vocabulary mapping for dummy data
        self.dummy_vocab = list(tokenizer.get_vocab().keys())
        self.dummy_vocab_size = len(self.dummy_vocab)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random sequence of token IDs
        input_ids = torch.randint(0, self.dummy_vocab_size, (self.sequence_length,), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long) # No padding for dummy data
        
        # For language modeling, target is usually the input shifted by one.
        # The last token has no target.
        # input_ids will be (seq_len,)
        # labels will be (seq_len,)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class DM(pl.LightningDataModule):
    def __init__(self, dataset_path: str, tokenizer_factory: callable, batch_size: int, 
                 sequence_length: int, num_workers: int, seed: int):
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer_factory = tokenizer_factory
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.seed = seed
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None # Not used in config, but common for DataModule

    def prepare_data(self):
        # Download/tokenize data if not already done.
        # This runs on a single GPU/process.
        self.tokenizer = self.tokenizer_factory()
        # Ensure tokenizer has a pad_token, as some models like gpt2 don't by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # You might need to resize embeddings in your model if vocab size changes

    def setup(self, stage: str):
        # Assign train/val/test datasets for use in dataloaders.
        # This runs on every GPU/process.
        # For "The Pile", you'd load actual data here.
        # For general purpose, we use dummy data.
        torch.manual_seed(self.seed) # For reproducible dummy data

        if stage == "fit" or stage == "train":
            self.train_dataset = DummyTextDataset(self.tokenizer, self.sequence_length, num_samples=10000)
            self.val_dataset = DummyTextDataset(self.tokenizer, self.sequence_length, num_samples=1000)
        elif stage == "validate":
            self.val_dataset = DummyTextDataset(self.tokenizer, self.sequence_length, num_samples=1000)
        # You'd typically split your actual dataset here after loading based on `dataset_path`

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            #pin_memory=True,
            pin_memory=torch.cuda.is_available(), # Improves data transfer to GPU
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            #pin_memory=True,
            pin_memory=torch.cuda.is_available()
        )

    # test_dataloader, predict_dataloader can be added similarly if needed
