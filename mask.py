# mask.py
import torch
import math

class AlibiMask(torch.nn.Module):
    def __init__(self, n_head: int, max_sequence_length: int, **kwargs):
        super().__init__()
        self.n_head = n_head
        self.max_sequence_length = max_sequence_length
        self.slopes = self._get_slopes(n_head)
        self.register_buffer("mask", self._create_alibi_mask(max_sequence_length, self.slopes))

    def _get_slopes(self, n_head):
        # Function to generate slopes for ALiBi from the original paper
        # m = 2 ** (-8 / n_head)
        # slopes = [m ** (i + 1) for i in range(n_head)]
        # This implementation uses the logarithmic approach for stability/precision
        def get_slopes_power_of_2(n_head):
            start = (2**(-8/n_head))
            ratio = start
            return [start * ratio**i for i in range(n_head)]
        
        if math.log2(n_head).is_integer():
            # Power of 2
            return get_slopes_power_of_2(n_head)
        else:
            # Not a power of 2, just take the first n_head slopes from a larger power of 2
            closest_power_of_2 = 2**math.ceil(math.log2(n_head))
            return get_slopes_power_of_2(closest_power_of_2)[:n_head]

    def _create_alibi_mask(self, seq_len: int, slopes: list):
        # Create a causal mask (lower triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * -torch.inf, diagonal=1)

        # Create distance matrix (relative positions)
        distance_matrix = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        
        # Expand slopes for broadcasting
        slopes = torch.tensor(slopes).float().unsqueeze(1).unsqueeze(1) # (n_head, 1, 1)

        # Alibi bias = slopes * distance_matrix
        # distance_matrix: (1, seq_len, seq_len)
        # alibi_bias: (n_head, seq_len, seq_len)
        alibi_bias = slopes * distance_matrix.unsqueeze(0)

        # Combine with causal mask
        # Add the causal mask to apply -inf to future tokens.
        # Adding -inf effectively makes those attention scores zero after softmax.
        final_mask = alibi_bias + causal_mask.unsqueeze(0) # (n_head, seq_len, seq_len)
        
        return final_mask

    def forward(self, query_len: int, key_len: int, device: torch.device):
        # In a typical setup, query_len and key_len might vary,
        # but for simplicity here we assume they match max_sequence_length
        # and slice the precomputed mask.
        if query_len > self.max_sequence_length or key_len > self.max_sequence_length:
            raise ValueError("Query or key length exceeds max_sequence_length for AlibiMask.")
        
        # Returns the mask slice for the current sequence length
        return self.mask[:, :query_len, :key_len].to(device)
