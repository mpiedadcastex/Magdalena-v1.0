import torch
import torch.nn as nn
from .encoder import AudioEncoder
from .decoder import MidiDecoder

class PianoTranscriptionModel(nn.Module):
    def __init__(self, config):
        """
        Modelo completo de Transcripción de Piano.
        Combina un Encoder de Audio (AudioEncoder) y un Decoder de MIDI (MidiDecoder).

        Args:
            config: Diccionario con hiperparametros (n_mels, d_model, vocab_size, ...).
        """
        super().__init__()

        # 1. Inicialización del Encoder
        self.encoder = AudioEncoder(
            n_mels=config['n_mels'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_encoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )

        # 2. Inicialización del Decoder
        self.decoder = MidiDecoder(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_decoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )

        # 3. Parámetros comunes
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']

    def encode(self, audio):
        """
        Paso 1: Procesar el audio para generar la memoria latente.
        
        """
        return self.encoder(audio)
    
    def decode(self, tgt, memory):
        """
        Paso 2: Generar logits a partir de la memoria del encoder y los tokens previos.
        
        """
        return self.decoder(tgt, memory)
    
    def forward(self, audio, tgt):
        """
        Flujo de entrenamiento completo estándar (Teacher Forcing).

        Args:
            audio: Espectrograma Mel [Batch, 1, n_mels, Time]
            tgt: Secuencia de tokens objetivo [Batch, Seq_Len] (entrada del decoder)

        Returns:
            logits: Probabilidades de cada token [Batch, Seq_Len, vocab_size]
        """
        # 1. Codificar el audio
        memory = self.encode(audio)  # [Batch, Audio_Len, d_model]

        # 2. Decodificar con la memoria y los tokens objetivo
        logits = self.decode(tgt, memory)  # [Batch, Seq_Len, vocab_size]

        return logits
    
    @torch.no_grad()
    def generate(self, audio, start_token, max_len=1000, end_token=None):
        """
        Inferencia (Greedy Decoding)
        Se usa cuando el modelo ya está entrenado y queremos transcribir un audio nuevo
        """
        self.eval() # Modo evaluación
        device = audio.device

        # 1. Codificar el audio
        memory = self.encode(audio)  # [1, Audio_Len, d_model]

        # 2. Empezar la frase con el token de inicio (start_token)
        # Iniciamos con batch size 1 conteniendo el token de inicio
        current_tokens = torch.tensor([[start_token]], dtype=torch.long, device=device)  

        generated_seq = []

        print("Generando transcripción...")

        # 3. Bucle de generación paso a paso
        for _ in range(max_len):
            # Obetener prediccion para el siguiente token
            logits = self.decode(current_tokens, memory)  # [1, Seq_Len, vocab_size]

            # Miramos solo el último paso
            last_token_logits = logits[:, -1, :]  # [1, vocab_size]

            # Token con mayor probabilidad (Greedy)
            next_token = torch.argmax(last_token_logits, dim=-1)  # [1]

            # Añadir a la secuencia generada
            generated_seq.append(next_token.item())

            # Añadir a la entrada para el siguiente paso
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)  # [1, Seq_Len+1]

            # Condición de parada (si se alcanza el token de fin)
            if end_token is not None and next_token.item() == end_token:
                break

        return generated_seq