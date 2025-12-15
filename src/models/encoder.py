import torch
import torch.nn as nn
import math
from einops import rearrange
from .layers.positional_encoding import PositionalEncoding

class AudioEncoder(nn.Module):
    def __init__(self, n_mels=229, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Encoder del Transformer para Audio.
        
        Args:
            n_mels (int): Altura del espectrograma (frecuencias).
            d_model (int): Dimensión interna del Transformer (embedding size).
            nhead (int): Número de cabezas de atención.
            num_layers (int): Número de capas del encoder.
            dim_feedforward (int): Tamaño de la capa densa intermedia.
        """
        super().__init__()
        self.model_type = 'TransformerEncoder'

        # 1. Capa de proyeccion de entrada
        # Transforma cada columna de frecuencias Mel a un vector d_model
        self.input_projection = nn.Linear(n_mels, d_model)

        # 2. Codificación posicional
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # 3. Capas del Transformer Encoder
        # Usamos la implementación estándar de PyTorch por ahora (Atención Global).
        # Más adelante, aquí es donde podrás intercambiar esto por tu 'Sparsifiner'.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.d_model = d_model


    def forward(self, src):
        """
        Args:
            src: Espectrograma Mel [Batch, 1, n_mels, time]
        Returns:
            memory: Representación latente del audio [Batch, time, d_model]
        """
        # A. Preparar dimensiones
        # Entrada original: [Batch, Channels=1, Freq=229, Time]
        # El Transformer espera una secuencia: [Batch, Time, Features]
        
        # Usamos einops para permutar y eliminar el canal mono
        # 'b c f t -> b t (c f)' fusiona canal y frecuencia
        x = rearrange(src, 'b c f t -> b t (c f)')
        
        # B. Proyección Lineal
        x = self.input_projection(x) # Ahora es [Batch, Time, d_model]
        
        # C. Escalar por raíz de d_model (Truco estándar de Transformers)
        x = x * math.sqrt(self.d_model)
        
        # D. Añadir Posición
        x = self.pos_encoder(x)
        
        # E. Pasar por las capas del Transformer
        memory = self.transformer_encoder(x)
        
        return memory