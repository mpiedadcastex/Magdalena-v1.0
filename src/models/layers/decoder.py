import torch
import torch.nn as nn
import math

from src.data.midi_proc import MidiProcessor
from .positional_encoding import PositionalEncoding

class MidiDecoder(nn.Module):
    def __init__(self, midi_processor: MidiProcessor, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        vocab_size =midi_processor.vocab_size
        
        # 1. Embedding de Entrada
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # 3. Bloques del Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. Capa de salida
        self.output_layer = nn.Linear(d_model, vocab_size)

        self.d_model = d_model


    def generate_square_subsequent_mask(self, sz, device):
        """
        Genera una máscara triangular superior con -inf.
        Asegura que la posición 'i' solo pueda atender a posiciones de 0 a i.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, tgt, memory, tgt_padding_mask=None):
        """
        Args:
            tgt: Secuencia de tokens objetivo [Batch, Seq_Len] (lo que ya se ha escrito)
            memory: Salida del Encoder [Batch, Audio_Len, d_model] (el "contexto" del audio)
            tgt_padding_mask: [Batch, Seq_Len] (Booleano: True donde es padding)
        """  
        # 1. Crear Máscara triangular
        seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len, tgt.device)

        # 2. Embedding y Positional Encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # 3. Pasar por el Transformer Decoder
        # -> Self-Attention con máscara (tokens previos)
        # -> Cross-Attention con la memoria del Encoder (memory)
        output = self.transformer_decoder(
            tgt = x,
            memory = memory,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # 4. Capa de Predicción
        logits = self.output_layer(output)  # [Batch, Seq_Len, vocab_size]

        return logits

