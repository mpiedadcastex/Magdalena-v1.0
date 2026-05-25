import re
import os
import time
import argparse
import importlib
from types import SimpleNamespace

# Configuración para evitar fragmentación de memoria en la GPU
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm # Barra de progreso
from torch.utils.tensorboard import SummaryWriter

# Tus módulos
from src.data.audio_proc import AudioProcessor
from src.data.midi_proc import MidiProcessor
from src.data.maestro_dataset import get_dataloaders
from src.models.transformer import PianoTranscriptionModel
from utils import get_model_config, log_training_loss
from evaluate import evaluate_model


# --- CONFIGURACIÓN ---
# Los hiperparámetros se cargan dinámicamente según el experimento:
#   Sin --exp  → v1_baseline (valores de producción definidos en load_config)
#   Con --exp  → experiments/<nombre>/config.py

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo de transcripción de piano.")
    parser.add_argument(
        '--exp', type=str, default=None,
        help='Nombre del experimento en experiments/. Si no se especifica, se usa v1_baseline.'
    )
    return parser.parse_args()


def load_config(exp_name=None):
    """
    Carga la configuración de producción (v1_baseline) o la de un experimento específico.

    Args:
        exp_name (str | None): Nombre de la carpeta del experimento, o None para producción.

    Returns:
        SimpleNamespace con todos los hiperparámetros del entrenamiento.
    """
    if exp_name is None:
        return SimpleNamespace(
            EXP_NAME="v1_baseline",
            EXP_DESCRIPTION="Entrenamiento de producción completo con dataset MAESTRO.",
            MAX_SAMPLES_TRAIN=None,
            MAX_SAMPLES_VAL=None,
            BATCH_SIZE=8,                  # Limitado por la atención del Sparsifiner
            GRAD_ACCUMULATION_STEPS=4,     # Batch efectivo = 32
            LEARNING_RATE=1e-4,            # Estándar para Transformers
            EPOCHS=80,
            EMBED_DIM=256,                 # 256 para probar, sube a 512 si tienes VRAM
            NUM_ENCODER_LAYERS=4,
            NUM_DECODER_LAYERS=4,
            NHEAD=4,
            CSV_PATH='/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv',
            ROOT_DIR='/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0',
            CHECKPOINT_DIR="checkpoints/v1_baseline",
            DRIVE_LOG_PATH="/content/drive/MyDrive/TFG_Project/MPCS/checkpoints/v1_baseline/"
        )

    module = importlib.import_module(f"experiments.{exp_name}.config")
    return SimpleNamespace(**{
        k: getattr(module, k)
        for k in dir(module)
        if not k.startswith('_')
    })


def train(cfg):
    DEVICE               = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_CHECKPOINT     = os.path.join(cfg.CHECKPOINT_DIR, "model")
    OPTIMIZER_CHECKPOINT = os.path.join(cfg.CHECKPOINT_DIR, "optimizer")

    print(f"[{cfg.EXP_NAME}] Usando dispositivo: {DEVICE}")

    # Si no existen las carpetas de guardado de checkpoints las creamos
    # MODELO
    os.makedirs(MODEL_CHECKPOINT, exist_ok=True)

    # OPTIMIZER
    os.makedirs(OPTIMIZER_CHECKPOINT, exist_ok=True)

    writer = SummaryWriter(log_dir=cfg.CHECKPOINT_DIR)

    # 1. PREPARAR DATOS
    print("Cargando datos...")
    ap = AudioProcessor(fmax=8000, n_mels=229)
    mp = MidiProcessor() # Asegúrate que tu MidiProcessor tenga vocab_size

    train_loader, val_loader = get_dataloaders(
        csv_path=cfg.CSV_PATH,      # <--- AJUSTA EN config.py
        root_dir=cfg.ROOT_DIR,      # <--- AJUSTA EN config.py
        audio_processor=ap,
        midi_processor=mp,
        batch_size=cfg.BATCH_SIZE,
        max_samples_train=cfg.MAX_SAMPLES_TRAIN,
        max_samples_val=cfg.MAX_SAMPLES_VAL,
        max_audio_frames=getattr(cfg, 'MAX_AUDIO_FRAMES', 4096),
        max_midi_tokens=getattr(cfg, 'MAX_MIDI_TOKENS', 1500)
    )

    # 2. PREPARAR MODELO
    print("Inicializando modelo...")
    model_cfg = get_model_config()

    vocab_size = mp.vocab_size

    model = PianoTranscriptionModel(
        midi_processor=mp,
        encoder_cfg=model_cfg,
        embed_dim=cfg.EMBED_DIM,
        num_encoder_layers=cfg.NUM_ENCODER_LAYERS,
        num_decoder_layers=cfg.NUM_DECODER_LAYERS,
        nhead=cfg.NHEAD
    ).to(DEVICE)

    # 3. OPTIMIZADOR Y PÉRDIDA
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    # ignore_index=0 es CRÍTICO para que no aprenda del padding
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # --- SISTEMA DE REANUDACIÓN ---
    START_EPOCH = 0

    print(f"Buscando checkpoints disponibles ...")

    model_files = [f for f in os.listdir(MODEL_CHECKPOINT) if f.endswith('.pth')]

    if model_files:
        epochs_found = []
        for f in model_files:
            match = re.search(r'model_epoch_(\d+).pth', f)
            if match:
                epochs_found.append(int(match.group(1)))

        if epochs_found:
            max_epoch = max(epochs_found)

            # Construimos las rutas de carga
            model_load_path = os.path.join(MODEL_CHECKPOINT, f"model_epoch_{max_epoch}.pth")
            optimizer_load_path = os.path.join(OPTIMIZER_CHECKPOINT, f"opt_epoch_{max_epoch}.pth")

            try:
                # carga de los pesos del modelo
                model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
                print(f"Modelo cargado: {model_load_path}")

                # Carga del estado del optimizador
                if os.path.exists(optimizer_load_path):
                    optimizer.load_state_dict(torch.load(optimizer_load_path, map_location=DEVICE))
                    print(f"Optimizador cargado: {optimizer_load_path}")
                else:
                    print(f"AVISO --> No se ha encontrado checkpoint para el optimizador. Se usará uno nuevo")

                # Ajustamos la epoch de inicio
                START_EPOCH = max_epoch
                print(f"Reanudando entrenamiento desde epoch {START_EPOCH + 1}")

            except Exception as e:
                print(f"Error al cargar el checkpoint {e}. Empezando desde cero.")
        else:
            print(f"No se encontraron archivos con el formato correcto. Empezando desde cero.")
    else:
        print(f"No se encontraron checkpoints. Empezando desde cero.")

    # --- BUCLE DE ENTRENAMIENTO ---
    optimizer.zero_grad()

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(START_EPOCH, cfg.EPOCHS):
        model.train()
        total_loss = 0
        epoch_start = time.time()

        loop = tqdm(train_loader, desc=f"[{cfg.EXP_NAME}] Epoch {epoch+1}/{cfg.EPOCHS}")

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
                loss = loss / cfg.GRAD_ACCUMULATION_STEPS

            # Backward
            scaler.scale(loss).backward()

            # Definimos las condiciones para hacer step
            is_accumulation_step = (i + 1) % cfg.GRAD_ACCUMULATION_STEPS == 0
            is_last_step = (i + 1) == len(train_loader)

            if is_accumulation_step or is_last_step:

                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Actualizar pesos con scaler
                scaler.step(optimizer)
                scaler.update()

                # AHORA LIMPIAMOS para el siguiente grupo de acumulación
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            current_loss = loss.item() * cfg.GRAD_ACCUMULATION_STEPS
            total_loss += current_loss
            loop.set_postfix(loss=current_loss)

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        epochs_remaining = cfg.EPOCHS - (epoch + 1)
        estimated_remaining = epochs_remaining * epoch_time

        print(f"Fin Epoch {epoch+1} | Loss: {avg_loss:.4f} | Tiempo: {epoch_time/60:.1f} min | Restante estimado: {estimated_remaining/3600:.1f}h")

        log_training_loss(epoch, avg_loss, log_interval=1, base_path=cfg.DRIVE_LOG_PATH)
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)
        # Definimos los nomrbes de los archivos de guardado
        current_epoch_save = epoch + 1

        model_filename = f"model_epoch_{current_epoch_save}.pth"
        optimizer_filename = f"opt_epoch_{current_epoch_save}.pth"

        model_save_path = os.path.join(MODEL_CHECKPOINT, model_filename)
        optimizer_save_path = os.path.join(OPTIMIZER_CHECKPOINT, optimizer_filename)

        print(f"Guardando checkpoints de epoch {current_epoch_save}...")

        try:
            # Guardamos los checkpoints en sus carpetas correspondientes
            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), optimizer_save_path)
            print(f"Guardado completado con éxito")
        except Exception as e:
            print(f"Error al guardar el checkpoint {e}")

        if (epoch + 1) % 20 == 0:
            evaluate_model(model, val_loader, mp, DEVICE, writer=writer, epoch=epoch)


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.exp)
    train(cfg)
