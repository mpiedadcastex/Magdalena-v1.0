"""
Genera retroactivamente las métricas F1 en TensorBoard para los checkpoints
ya guardados de un experimento.

Compara los epochs con checkpoint disponible (múltiplos de EVAL_INTERVAL) contra
los epochs que ya tienen entradas F1 en el archivo de TensorBoard. Evalúa solo
los que faltan, añadiendo los escalares al mismo log_dir para que TensorBoard
muestre una serie temporal completa.

Uso:
    python retroactive_eval.py                          # v1_baseline
    python retroactive_eval.py --exp exp01_reduced_data
"""

import os
import re
import argparse
import importlib
from types import SimpleNamespace

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.data.audio_proc import AudioProcessor
from src.data.midi_proc import MidiProcessor
from src.data.maestro_dataset import get_dataloaders
from src.models.transformer import PianoTranscriptionModel
from utils import get_model_config
from evaluate import evaluate_model


EVAL_INTERVAL = 5  # Debe coincidir con el intervalo usado en train.py


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(exp_name=None):
    if exp_name is None:
        return SimpleNamespace(
            EXP_NAME="v1_baseline",
            MAX_SAMPLES_TRAIN=None,
            MAX_SAMPLES_VAL=None,
            EMBED_DIM=256,
            NUM_ENCODER_LAYERS=4,
            NUM_DECODER_LAYERS=4,
            NHEAD=4,
            CSV_PATH='/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv',
            ROOT_DIR='/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0',
            CHECKPOINT_DIR="checkpoints/v1_baseline",
        )
    module = importlib.import_module(f"experiments.{exp_name}.config")
    return SimpleNamespace(**{k: getattr(module, k) for k in dir(module) if not k.startswith('_')})


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------

def get_logged_f1_epochs(log_dir):
    """
    Devuelve el conjunto de steps (= epoch_num) que ya tienen
    métricas F1/onset registradas en el directorio de TensorBoard.
    """
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        if 'F1/onset' not in tags:
            return set()
        return {int(e.step) for e in ea.Scalars('F1/onset')}
    except Exception as exc:
        print(f"  [AVISO] No se pudo leer el archivo de TensorBoard: {exc}")
        return set()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def get_eval_checkpoints(checkpoint_dir, eval_interval):
    """
    Devuelve lista ordenada de epoch_num que tienen checkpoint guardado
    y son múltiplos de eval_interval (los que debería haber evaluado train.py).
    """
    model_dir = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(model_dir):
        return []
    epochs = []
    for fname in os.listdir(model_dir):
        match = re.search(r'model_epoch_(\d+)\.pth', fname)
        if match:
            epoch = int(match.group(1))
            if epoch % eval_interval == 0:
                epochs.append(epoch)
    return sorted(epochs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generación retroactiva de métricas F1 para checkpoints ya guardados."
    )
    parser.add_argument(
        '--exp', type=str, default=None,
        help='Nombre del experimento en experiments/. Sin argumento = v1_baseline.'
    )
    args = parser.parse_args()

    cfg = load_config(args.exp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n[{cfg.EXP_NAME}] Dispositivo: {device}")
    print(f"Directorio de checkpoints : {cfg.CHECKPOINT_DIR}")
    print(f"Intervalo de evaluación   : cada {EVAL_INTERVAL} epochs\n")

    # 1. Epochs con checkpoint guardado que tocan evaluación
    eval_epochs = get_eval_checkpoints(cfg.CHECKPOINT_DIR, EVAL_INTERVAL)
    if not eval_epochs:
        print("No se encontraron checkpoints en el directorio. Abortando.")
        return
    print(f"Checkpoints evaluables encontrados : {eval_epochs}")

    # 2. Epochs que ya tienen métricas F1 en TensorBoard
    logged_epochs = get_logged_f1_epochs(cfg.CHECKPOINT_DIR)
    print(f"Epochs ya registrados en TensorBoard: {sorted(logged_epochs)}")

    # 3. Calcular los que faltan
    missing_epochs = [e for e in eval_epochs if e not in logged_epochs]
    if not missing_epochs:
        print("\nTodo está al día. No hay métricas que generar.")
        return
    print(f"Epochs pendientes de evaluar : {missing_epochs}\n")

    # 4. Preparar datos (batch_size=1 para evitar batch mismatch en predict_sampling)
    print("Cargando datos de validación...")
    ap = AudioProcessor(fmax=8000, n_mels=229)
    mp = MidiProcessor()

    _, val_loader = get_dataloaders(
        csv_path=cfg.CSV_PATH,
        root_dir=cfg.ROOT_DIR,
        audio_processor=ap,
        midi_processor=mp,
        batch_size=1,
        max_samples_train=cfg.MAX_SAMPLES_TRAIN,
        max_samples_val=cfg.MAX_SAMPLES_VAL,
    )

    # 5. Preparar modelo
    print("Inicializando modelo...")
    model_cfg = get_model_config()
    model = PianoTranscriptionModel(
        midi_processor=mp,
        encoder_cfg=model_cfg,
        embed_dim=cfg.EMBED_DIM,
        num_encoder_layers=cfg.NUM_ENCODER_LAYERS,
        num_decoder_layers=cfg.NUM_DECODER_LAYERS,
        nhead=cfg.NHEAD,
    ).to(device)

    # El writer se abre sobre el mismo log_dir para añadir al historial existente
    writer = SummaryWriter(log_dir=cfg.CHECKPOINT_DIR)

    # 6. Evaluar cada epoch pendiente
    for epoch_num in missing_epochs:
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "model", f"model_epoch_{epoch_num}.pth")
        print(f"\n{'='*60}")
        print(f"Evaluando epoch {epoch_num} | Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        except Exception as exc:
            print(f"  [ERROR] No se pudo cargar el checkpoint: {exc}. Saltando.")
            continue

        # evaluate_model usa epoch+1 como step de TensorBoard, así que pasamos epoch_num-1
        evaluate_model(model, val_loader, mp, device, writer=writer, epoch=epoch_num - 1)

    writer.close()
    print("\nGeneración de métricas retroactiva completada.")


if __name__ == "__main__":
    main()
