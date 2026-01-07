import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=50000): 
        super().__init__()                                   
        self.dropout = nn.Dropout(p=dropout)

        # ---------------------------------------------------------------------
        # 1. PREPARACIÓN MATEMÁTICA (frecuencias)
        # ---------------------------------------------------------------------
        # El objetivo es crear ondas de diferentes frecuencias.
        # position: Vector columna [0, 1, 2, ..., max_len-1].
        # Representa "t" en las fórmulas. Shape: (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # div_term: Calculamos el denominador de la fórmula geométrica.
        # Fórmula original: 1 / (10000 ^ (2i / d_model))
        # Implementación logarítmica (más estable numéricamente): exp(2i * -log(10000)/d_model)
        # Esto genera un vector de frecuencias decrecientes.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # ---------------------------------------------------------------------
        # 2. CONSTRUCCIÓN DE LA MATRIZ PE
        # ---------------------------------------------------------------------
        # Inicializamos matriz de ceros: (max_len, 1, d_model)
        # La dimensión '1' en el medio es para facilitar el "broadcasting" luego.
        pe = torch.zeros(max_len, 1, d_model)

        # Llenamos las posiciones PARES (0::2) con SENO
        # pe[pos, 2i] = sin(pos / 10000^(2i/d_model))
        # Al multiplicar (max_len, 1) * (d_model/2), PyTorch expande las dimensiones automáticamente.
        pe[:, 0, 0::2] = torch.sin(position * div_term)

        # Llenamos las posiciones IMPARES (1::2) con COSENO
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # ---------------------------------------------------------------------
        # 2. CONSTRUCCIÓN DE LA MATRIZ PE
        # ---------------------------------------------------------------------
        # Registramos como buffer (se guarda con el modelo pero no se entrena)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor de entrada [Batch, Secuencia, Features]
        """
        # x.size(1) es la longitud de la secuencia actual (ej. 100 tokens).
        # self.pe tiene un tamaño máximo (ej. 50000), así que cortamos lo que necesitamos.
        
        # El tensor 'pe' original tiene forma: (Max_Len, 1, Dim)
        # El tensor 'x' tiene forma:          (Batch, Seq_Len, Dim)
        
        # Para sumarlos, necesitamos que 'pe' tenga forma: (1, Seq_Len, Dim)

        
        if x.size(1) == self.pe.size(0): 
             # Caso raro: La secuencia es tan larga como el max_len (o x no es batch_first)
             x = x + self.pe[:x.size(1), :].permute(1, 0, 2)

        elif x.size(0) == self.pe.size(0): 
             # Caso raro: La dimensión 0 coincide con max_len
             x = x + self.pe[:x.size(0), :]

        else:
             # --- CASO COMÚN (Batch First) ---
            # 1. self.pe[:x.size(1), :]  -> Cortamos hasta la longitud actual. Queda (Seq, 1, Dim)
            # 2. .transpose(0, 1)        -> Giramos dimensiones 0 y 1. Queda (1, Seq, Dim)
            # 3. Suma: (Batch, Seq, Dim) + (1, Seq, Dim) -> El PE se suma a cada ejemplo del batch.
            x = x + self.pe[:x.size(1), :].transpose(0, 1)

        return self.dropout(x)