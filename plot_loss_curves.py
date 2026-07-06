"""
plot_loss_curves.py — Genera gráficas de curvas de pérdida de entrenamiento
a partir de los eventos de TensorBoard guardados en las carpetas de checkpoints.

Salida (carpeta plots/):
  - loss_v1_baseline.png / .svg
  - loss_exp01_reduced_data.png / .svg
  - loss_exp02_short_clips.png / .svg
  - loss_all_experiments.png / .svg

Uso: python plot_loss_curves.py [--exp NOMBRE]
  Sin --exp  → genera todos los plots
  Con --exp  → genera solo el del experimento indicado (+ el combinado)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Sin GUI; compatible con Colab y entornos headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ─── Rutas ────────────────────────────────────────────────────────────────────
_IN_COLAB   = os.path.isdir('/content/drive')
_DRIVE_BASE = '/content/drive/MyDrive/TFG_Project/MPCS/checkpoints'
_LOCAL_BASE = 'checkpoints'

PLOTS_DIR   = 'plots'

# ─── Configuración de experimentos ────────────────────────────────────────────
EXPERIMENTS = [
    {
        'name':  'v1_baseline',
        'label': 'v1_baseline (épocas 52–100)',
        'color': '#2196F3',   # azul
    },
    {
        'name':  'exp01_reduced_data',
        'label': 'exp01 — dataset reducido',
        'color': '#FF9800',   # naranja
    },
    {
        'name':  'exp02_short_clips',
        'label': 'exp02 — clips cortos',
        'color': '#4CAF50',   # verde
    },
]

# ─── Estilo visual ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'legend.fontsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
})


def load_scalars(tb_dir, tag='Loss/train'):
    """
    Lee eventos de TensorBoard y devuelve (epochs, losses) como arrays numpy.
    Devuelve (None, None) si no hay datos o el directorio no existe.
    """
    if not os.path.isdir(tb_dir):
        return None, None

    ea = EventAccumulator(tb_dir, size_guidance={'scalars': 0})
    ea.Reload()

    if tag not in ea.Tags().get('scalars', []):
        return None, None

    events = ea.Scalars(tag)
    epochs = np.array([e.step  for e in events])
    losses = np.array([e.value for e in events])

    # Ordenar por epoch por si acaso
    order  = np.argsort(epochs)
    return epochs[order], losses[order]


def save_fig(fig, name):
    """Guarda la figura en PNG y SVG."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for ext in ('png', 'svg'):
        path = os.path.join(PLOTS_DIR, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight')
        print(f"  Guardado: {path}")


def plot_single(exp, base):
    """Genera el plot individual de un experimento."""
    tb_dir = os.path.join(base, exp['name'])
    epochs, losses = load_scalars(tb_dir)

    if epochs is None:
        print(f"  Sin datos para {exp['name']} en {tb_dir}")
        return None, None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, losses, color=exp['color'], linewidth=1.8, label=exp['label'])

    ax.set_title(f"Curva de pérdida — {exp['label']}")
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss (entrenamiento)')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()

    save_fig(fig, f"loss_{exp['name']}")
    plt.close(fig)
    return epochs, losses


def plot_combined(all_data):
    """Genera el plot con todos los experimentos superpuestos."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for exp, (epochs, losses) in zip(EXPERIMENTS, all_data):
        if epochs is None:
            continue
        ax.plot(epochs, losses, color=exp['color'], linewidth=1.8, label=exp['label'])

    ax.set_title('Comparativa de curvas de pérdida — todos los experimentos')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss (entrenamiento)')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()

    save_fig(fig, 'loss_all_experiments')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Genera plots de curvas de pérdida desde eventos TensorBoard")
    parser.add_argument(
        '--exp', type=str, default=None,
        help='Nombre del experimento. Sin --exp genera todos.'
    )
    args = parser.parse_args()

    base    = _DRIVE_BASE if _IN_COLAB else _LOCAL_BASE
    targets = [e for e in EXPERIMENTS if args.exp is None or e['name'] == args.exp]

    if not targets:
        print(f"Experimento desconocido: '{args.exp}'")
        print(f"Opciones: {[e['name'] for e in EXPERIMENTS]}")
        return

    print(f"Leyendo eventos desde: {base}")
    print(f"Guardando plots en:    {PLOTS_DIR}/\n")

    all_data = []
    for exp in EXPERIMENTS:
        if exp in targets:
            print(f"-> {exp['name']}")
            epochs, losses = plot_single(exp, base)
            all_data.append((epochs, losses))
        else:
            # Cargar igualmente para el plot combinado
            tb_dir = os.path.join(base, exp['name'])
            all_data.append(load_scalars(tb_dir))

    print(f"\n-> Combinado")
    plot_combined(all_data)
    print("\nListo.")


if __name__ == '__main__':
    main()
