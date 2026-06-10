import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LengthPredictor(nn.Module):
    def __init__(self, d_model: int, num_length_bins: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_length_bins)
        self.num_length_bins = num_length_bins

    def forward(self, encoder_output: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        mask = ~src_padding_mask.unsqueeze(-1)
        masked_encoder_output = encoder_output.permute(1, 0, 2) * mask
        summed = masked_encoder_output.sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pooled = summed / valid_counts
        length_logits = self.linear(mean_pooled)
        return length_logits