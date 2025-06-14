# model/hparams.py

class HParams:
    def __init__(self, 
                 vocab_size: int,
                 max_sequence_length: int,
                 n_layer: int,
                 n_head: int,
                 d_model: int,
                 n_kv_head_ratio: float,
                 d_qk_ratio: float,
                 d_v_ratio: float,
                 dropout: float,
                 feedforward_d_model_ratio: float):
        
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.n_kv_head_ratio = n_kv_head_ratio
        self.d_qk_ratio = d_qk_ratio
        self.d_v_ratio = d_v_ratio
        self.dropout = dropout
        self.feedforward_d_model_ratio = feedforward_d_model_ratio

        # Calculated parameters (for convenience)
        self.d_head = d_model // n_head
        self.d_qk = int(self.d_head * d_qk_ratio)
        self.d_v = int(self.d_head * d_v_ratio)
        self.n_kv_head = int(n_head * n_kv_head_ratio)
        self.d_feedforward = int(d_model * feedforward_d_model_ratio)

    def __repr__(self):
        return (f"HParams(vocab_size={self.vocab_size}, max_sequence_length={self.max_sequence_length}, "
                f"n_layer={self.n_layer}, n_head={self.n_head}, d_model={self.d_model}, "
                f"dropout={self.dropout})")
