import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # CAMBIO 1: Aumentar max_len (original 5000) para evitar problemas con secuencias largas
    def __init__(self, d_model, dropout=0.1, max_len=50000): 
        super().__init__()                                   
        self.dropout = nn.Dropout(p=dropout)

        # Crear una matriz de posiciones y dimensiones
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)

        # Aplicar las fórmulas de codificación posicional
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Registramos como buffer (se guarda con el modelo pero no se entrena)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor de entrada [Batch, Secuencia, Features]
        """
        # Sumamos la codificación posicional hasta la longitud actual de x
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)