import os
# Configuración para evitar fragmentación de memoria en la GPU
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm # Barra de progreso

# Tus módulos
from src.data.audio_proc import AudioProcessor
from src.data.midi_proc import MidiProcessor
from src.data.maestro_dataset import get_dataloaders
from src.models.transformer import PianoTranscriptionModel
from utils import get_model_config

# --- HIPERPARÁMETROS ---
BATCH_SIZE = 1         # Pequeño por la VRAM de Colab (con 2 ha explotado)
GRAD_ACCUMULATION_STEPS = 4  # Simulamos batch de 4
LEARNING_RATE = 1e-4   # Estándar para Transformers
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    print(f"Usando dispositivo: {DEVICE}")

    # 1. PREPARAR DATOS
    print("Cargando datos...")
    ap = AudioProcessor(fmax=8000, n_mels=229)
    mp = MidiProcessor() # Asegúrate que tu MidiProcessor tenga vocab_size
    
    train_loader, val_loader = get_dataloaders(
        csv_path='/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv',      # <--- AJUSTA LA RUTA
        root_dir='/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0',     # <--- AJUSTA LA RUTA
        audio_processor=ap, 
        midi_processor=mp, 
        batch_size=BATCH_SIZE
    )

    # 2. PREPARAR MODELO
    print("Inicializando modelo...")
    cfg = get_model_config()
    
    # IMPORTANTE: Obtener vocab_size de tu encoder o ponerlo a mano
    vocab_size = mp.vocab_size 
    
    model = PianoTranscriptionModel(
        midi_processor=mp,
        encoder_cfg=cfg,
        embed_dim=256,       # 256 para probar, sube a 512 si tienes VRAM
        num_encoder_layers=4,
        num_decoder_layers=4,
        nhead=4
    ).to(DEVICE)

    # 3. OPTIMIZADOR Y PÉRDIDA
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # ignore_index=0 es CRÍTICO para que no aprenda del padding
    criterion = nn.CrossEntropyLoss(ignore_index=0) 

    # --- BUCLE DE ENTRENAMIENTO ---
    optimizer.zero_grad()

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (batch_audio, batch_midi) in enumerate(loop):
            # batch_audio: (B, 229, Time)
            # batch_midi: (B, Seq_Len)
            
            batch_audio = batch_audio.to(DEVICE)
            batch_midi = batch_midi.to(DEVICE)

            # --- PREPARACIÓN DE TARGETS (TEACHER FORCING) ---
            # Input Decoder: Toda la secuencia MENOS el último token
            # Target (Lo que debe predecir): Toda la secuencia DESDE el segundo token
            # Ejemplo: [Start, Nota1, Nota2, End]
            # Input:   [Start, Nota1, Nota2]
            # Target:         [Nota1, Nota2, End]
            
            decoder_input = batch_midi[:, :-1]
            targets = batch_midi[:, 1:]

            # Máscara de Padding para que el decoder ignore los ceros del input
            # (True donde hay padding, False donde hay datos)
            tgt_padding_mask = (decoder_input == 0).to(DEVICE)

            # Forward
            # optimizer.zero_grad() -> Lo borramos porque vamos a arrastrar el gradiente entre batches
            
            with torch.amp.autocast('cuda'):
                logits = model(
                    src_audio=batch_audio, 
                    tgt_midi=decoder_input,
                    tgt_padding_mask=tgt_padding_mask
                )
                # Logits: (B, Seq_Len-1, Vocab)

                # Flatten para CrossEntropy
                # Reshape a (Batch * Seq_Len, Vocab) vs (Batch * Seq_Len)
                loss = criterion(
                    logits.reshape(-1, vocab_size), 
                    targets.reshape(-1)
                )

                # Normalizamos la pérdida para mantener la escala correcta
                loss = loss / GRAD_ACCUMULATION_STEPS

            # Backward
            scaler.scale(loss).backward()
            
            # Si ya hemos dado el nº de pasos establecido
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Actualizar pesos con scaler
                scaler.step(optimizer)
                scaler.update()
                
                # AHORA LIMPIAMOS para el siguiente grupo de acumulación
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            current_loss = loss.item() * GRAD_ACCUMULATION_STEPS
            total_loss += current_loss
            loop.set_postfix(loss=current_loss)

        avg_loss = total_loss / len(train_loader)
        print(f"Fin Epoch {epoch+1} - Loss Promedio: {avg_loss:.4f}")
        
        # Guardar checkpoint cada época
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()