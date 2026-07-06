"""
rebuild_tensorboard.py — Reconstruye los archivos de TensorBoard a partir de
training_loss_log.txt generado por log_training_loss() durante el entrenamiento.

No requiere GPU ni recargar el modelo.
Uso: python rebuild_tensorboard.py [--exp NOMBRE]

  Sin --exp  → reconstruye v1_baseline
  Con --exp  → reconstruye el experimento indicado (ej: --exp exp01_reduced_data)
"""

import re
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

# ─── Rutas por entorno ────────────────────────────────────────────────────────
_IN_COLAB = os.path.isdir('/content/drive')

# Raíz de checkpoints en Drive y local
_DRIVE_BASE  = '/content/drive/MyDrive/TFG_Project/MPCS/checkpoints'
_LOCAL_BASE  = 'checkpoints'

# Carpetas de cada experimento (nombre de carpeta en Drive / local)
_EXP_DIRS = {
    'v1_baseline':        'v1_baseline',
    'exp01_reduced_data': 'exp01',
    'exp02_short_clips':  'exp02',
}

# ─── Parsing del log ──────────────────────────────────────────────────────────
# Formato de cada fila: | 2024-01-01 12:00:00  | Epoch:    1    | Loss: 0.123456     |
_ROW_RE = re.compile(r'\|\s.*?\|\s*Epoch:\s*(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|')


def parse_loss_log(log_path):
    """
    Lee training_loss_log.txt y devuelve lista de (epoch, loss) ordenada por epoch.
    """
    entries = []
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            m = _ROW_RE.match(line)
            if m:
                epoch = int(m.group(1))
                loss  = float(m.group(2))
                entries.append((epoch, loss))

    entries.sort(key=lambda x: x[0])
    return entries


def rebuild(exp_name):
    exp_folder = _EXP_DIRS.get(exp_name)
    if exp_folder is None:
        print(f"Experimento desconocido: '{exp_name}'")
        print(f"Opciones disponibles: {list(_EXP_DIRS.keys())}")
        return

    # Ruta al archivo de log
    base     = _DRIVE_BASE if _IN_COLAB else _LOCAL_BASE
    log_path = os.path.join(base, exp_folder, 'training_loss_log.txt')

    if not os.path.exists(log_path):
        print(f"No se encontró el archivo de log en:\n  {log_path}")
        return

    print(f"Leyendo log: {log_path}")
    entries = parse_loss_log(log_path)

    if not entries:
        print("No se encontraron entradas en el log. Revisa el formato del archivo.")
        return

    print(f"Epochs encontradas: {len(entries)} (de {entries[0][0]} a {entries[-1][0]})")

    # Directorio de salida para TensorBoard
    tb_dir = os.path.join('runs', exp_name)
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)
    for epoch, loss in entries:
        writer.add_scalar('Loss/train', loss, epoch)
    writer.close()

    print(f"Archivos TensorBoard escritos en: {tb_dir}")
    print(f"Para visualizar ejecuta:\n  tensorboard --logdir runs/")


def main():
    parser = argparse.ArgumentParser(description="Reconstruye curvas TensorBoard desde training_loss_log.txt")
    parser.add_argument(
        '--exp', type=str, default='v1_baseline',
        help='Nombre del experimento. Por defecto: v1_baseline'
    )
    args = parser.parse_args()
    rebuild(args.exp)


if __name__ == '__main__':
    main()
