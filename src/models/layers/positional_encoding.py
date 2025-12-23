import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # CAMBIO 1: Aumentar max_len (original 5000) para evitar problemas con secuencias largas
    def __init__(self, d_model, max_len=10000, dropout=0.1): 
        super().__init__()                                   
        self.dropout = nn.Dropout(p=dropout)

        # Crear una matriz de posiciones y dimensiones
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Aplicar las fórmulas de codificación posicional
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Añadir dimensión de batch: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Registramos como buffer (se guarda con el modelo pero no se entrena)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor de entrada [Batch, Secuencia, Features]
        """
        # Sumamos la codificación posicional hasta la longitud actual de x
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            x = x[:, :self.pe.size(1), :]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)