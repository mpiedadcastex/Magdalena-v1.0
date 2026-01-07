import torch
import torch.nn as nn
from types import SimpleNamespace

from src.data.midi_proc import MidiProcessor

from .layers.sparsifiner_encoder import AudioSparsifinerEncoder
from .layers.decoder import MidiDecoder


class PianoTranscriptionModel(nn.Module):
    def __init__(
        self, 
        midi_processor: MidiProcessor,
        encoder_cfg: SimpleNamespace,
        embed_dim=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_audio_len=70000
    ):
        
        super().__init__()

        # 1. ENCODER SPARSIFINER
        self.encoder = AudioSparsifinerEncoder(
            mel_bins=229,
            max_seq_len=max_audio_len,
            embed_dim=embed_dim,
            depth=num_encoder_layers,
            num_heads=nhead,
            reduce_n_factor=64,     # Factor alto para ahorrar memoria
            attn_keep_rate=0.25,
            drop_rate=dropout,
            cfg=encoder_cfg         # Aquí es donde le pasamos la configuracion
        )

        # 2. DECODER
        self.decoder = MidiDecoder(
            midi_processor=midi_processor,
            d_model=embed_dim,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        

    def forward(self, src_audio, tgt_midi, tgt_mask=None, tgt_padding_mask=None):
        # 1. Codificar el src_audio
        memory = self.encoder(src_audio)  # [Batch, Audio_Len, d_model]

        # 2. Preparar MIDI
        logits = self.decoder(
            tgt=tgt_midi,
            memory=memory,
            tgt_padding_mask=tgt_padding_mask
        )

        return logits
    
    @torch.no_grad()
    def generate(self, src_audio, start_token, max_len=1000, end_token=None):
        """
        Inferencia (Greedy Decoding)
        Se usa cuando el modelo ya está entrenado y queremos transcribir un src_audio nuevo
        """
        self.eval() # Modo evaluación
        device = src_audio.device

        # 1. Codificar el src_audio
        memory = self.encoder(src_audio)  # [1, Audio_Len, d_model]

        # 2. Empezar la frase con el token de inicio (start_token)
        # Iniciamos con batch size 1 conteniendo el token de inicio
        current_tokens = torch.tensor([[start_token]], dtype=torch.long, device=device)  

        generated_seq = []

        print("Generando transcripción...")

        # 3. Bucle de generación paso a paso
        for _ in range(max_len):
            # Obetener prediccion para el siguiente token
            logits = self.decoder(current_tokens, memory)  # [1, Seq_Len, vocab_size]

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