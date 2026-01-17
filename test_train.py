import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# --- IMPORTA TU MODELO AQUÍ ---
# Asegúrate de que VS Code encuentra estas rutas
from src.models.transformer import PianoTranscriptionModel
from src.data.midi_proc import MidiProcessor
from utils import get_model_config

def test_training_loop():
    print("=== INICIANDO TEST DE ENTRENAMIENTO (DUMMY DATA) ===")
    
    # 1. Configuración Dummy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2 # Probamos con 2 para ver si aguanta
    seq_len = 100
    n_mels = 229
    vocab_size = 1000 # Simulado
    grad_acc_steps = 2 # Simulado
    
    # 2. Crear Datos Falsos
    # Audio: (Num_Muestras, n_mels, tiempo) -> simulamos tiempo=500 frames
    dummy_audio = torch.randn(10, n_mels, 500) 
    # Midi: (Num_Muestras, seq_len) -> enteros aleatorios entre 1 y vocab_size
    dummy_midi = torch.randint(1, vocab_size, (10, seq_len))
    
    # Dataset y Loader falso
    dataset = TensorDataset(dummy_audio, dummy_midi)
    loader = DataLoader(dataset, batch_size=batch_size)

    # 3. Inicializar Modelo (Usando tus clases reales si es posible, o mocks)
    print("Inicializando modelo...")
    
    # MOCK del MidiProcessor solo para obtener vocab_size si es necesario
    class MockMidiProc:
        vocab_size = 1000
    mp = MockMidiProc()
    
    try:
        cfg = get_model_config()
        model = PianoTranscriptionModel(
            midi_processor=mp,
            encoder_cfg=cfg,
            embed_dim=128,    # Dimensión reducida para el test
            num_encoder_layers=2,
            num_decoder_layers=2,
            nhead=4
        ).to(device)
    except Exception as e:
        print(f"❌ Error al inicializar tu modelo: {e}")
        print("Asegúrate de que las importaciones son correctas.")
        return

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler('cuda')

    print("Comenzando bucle de entrenamiento simulado...")
    
    model.train()
    
    try:
        for i, (batch_audio, batch_midi) in enumerate(loader):
            batch_audio = batch_audio.to(device)
            batch_midi = batch_midi.to(device)
            
            decoder_input = batch_midi[:, :-1]
            targets = batch_midi[:, 1:]
            tgt_padding_mask = (decoder_input == 0).to(device)

            # --- TU LÓGICA DE TRAIN ---
            with torch.amp.autocast('cuda'):
                logits = model(batch_audio, decoder_input, tgt_padding_mask=tgt_padding_mask)
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                loss = loss / grad_acc_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            print(f"   Batch {i+1} procesado. Loss: {loss.item() * grad_acc_steps:.4f}")
            
        print("\n✅ ¡ÉXITO! El bucle de entrenamiento funciona correctamente.")
        print("   - Las dimensiones coinciden.")
        print("   - El forward/backward pass no da error.")
        print("   - La acumulación de gradientes se ejecuta.")

    except RuntimeError as e:
        print(f"\n❌ ERROR DE EJECUCIÓN (Probablemente Shapes o Memoria):")
        print(e)
    except Exception as e:
        print(f"\n❌ ERROR GENÉRICO:")
        print(e)

if __name__ == "__main__":
    test_training_loop()