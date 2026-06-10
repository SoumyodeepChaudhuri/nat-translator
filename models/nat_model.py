import math
from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModel
from models.components import PositionalEncoding, LengthPredictor

class NATransformer(nn.Module):
    def __init__(self,
                 tgt_vocab_size: int,
                 d_model: int,
                 nhead: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 max_len: int = 5000,
                 pretrained_encoder_name: str = "facebook/mbart-large-50-many-to-many-mmt",
                 num_length_bins: int = 21):
        super().__init__()
        self.d_model = d_model

        print(f"Loading pretrained encoder: {pretrained_encoder_name}")
        try:
            pretrained_model = AutoModel.from_pretrained(pretrained_encoder_name)
            self.encoder = pretrained_model.encoder
        except Exception as e:
            print(f"Failed to load pretrained model {pretrained_encoder_name}: {e}")
            raise
        
        print("Pretrained encoder loaded.")

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False, activation='relu'
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.length_predictor = LengthPredictor(d_model, num_length_bins)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self,
                src_input_ids: torch.Tensor,
                src_attention_mask: torch.Tensor,
                tgt_input_ids: torch.Tensor,
                tgt_padding_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        encoder_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            return_dict=True
        )
        memory = encoder_outputs.last_hidden_state
        memory = memory.permute(1, 0, 2)
        memory_key_padding_mask = (src_attention_mask == 0)

        length_logits = self.length_predictor(memory, memory_key_padding_mask)
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt_input_ids) * math.sqrt(self.d_model))

        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        logits = self.output_projection(output)
        return logits, length_logits