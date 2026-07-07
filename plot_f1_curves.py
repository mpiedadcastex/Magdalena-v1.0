"""
plot_f1_curves.py — Genera gráficas de las métricas F1 de validación
a partir de los eventos de TensorBoard guardados en las carpetas de checkpoints.

Salida (results/plots/F1/):
  - f1_v1_baseline.png / .svg
  - f1_exp01_reduced_data.png / .svg
  - f1_exp02_short_clips.png / .svg
  - f1_all_experiments.png / .svg  (3 subplots, uno por métrica)

Uso: python plot_f1_curves.py [--exp NOMBRE]
  Sin --exp  → genera todos los plots
  Con --exp  → genera solo el del experimento indicado (+ el combinado)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ─── Rutas ────────────────────────────────────────────────────────────────────
_IN_COLAB   = os.path.isdir('/content/drive')
_DRIVE_BASE = '/content/drive/MyDrive/TFG_Project/MPCS/checkpoints'
_LOCAL_BASE = 'checkpoints'

PLOTS_DIR = os.path.join('results', 'plots', 'F1')

# ─── Experimentos ─────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        'name':  'v1_baseline',
        'label': 'v1_baseline',
        'color': '#2196F3',
    },
    {
        'name':  'exp01_reduced_data',
        'label': 'exp01 — dataset reducido',
        'color': '#FF9800',
    },
    {
        'name':  'exp02_short_clips',
        'label': 'exp02 — clips cortos',
        'color': '#4CAF50',
    },
]

# Métricas F1 disponibles en TensorBoard
F1_METRICS = [
    {'tag': 'F1/onset',          'label': 'F1 Onset'},
    {'tag': 'F1/onset_offset',   'label': 'F1 Onset + Offset'},
    {'tag': 'F1/velocity',       'label': 'F1 Onset + Velocity'},
]

# ─── Estilo visual ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'legend.fontsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
})

# Estilo de línea por métrica (dentro del plot individual)
_METRIC_STYLES = [
    {'linestyle': '-',  'linewidth': 1.8},
    {'linestyle': '--', 'linewidth': 1.6},
    {'linestyle': ':',  'linewidth': 1.8},
]


def load_scalars(tb_dir, tag):
    """
    Lee un tag de TensorBoard y devuelve (steps, values) como arrays numpy.
    Devuelve (None, None) si no hay datos o el directorio no existe.
    """
    if not os.path.isdir(tb_dir):
        return None, None

    ea = EventAccumulator(tb_dir, size_guidance={'scalars': 0})
    ea.Reload()

    if tag not in ea.Tags().get('scalars', []):
        return None, None

    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    order  = np.argsort(steps)
    return steps[order], values[order]


def save_fig(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for ext in ('png', 'svg'):
        path = os.path.join(PLOTS_DIR, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight')
        print(f"  Guardado: {path}")


def plot_single(exp, base):
    """
    Plot individual de un experimento: las 3 métricas F1 sobre el mismo eje.
    Devuelve dict {tag: (steps, values)} para reutilizar en el combinado.
    """
    tb_dir = os.path.join(base, exp['name'])
    data   = {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    has_data = False

    for metric, style in zip(F1_METRICS, _METRIC_STYLES):
        steps, values = load_scalars(tb_dir, metric['tag'])
        data[metric['tag']] = (steps, values)
        if steps is None:
            continue
        has_data = True
        ax.plot(steps, values, color=exp['color'],
                label=metric['label'], **style)

    if not has_data:
        print(f"  Sin eventos F1 para {exp['name']} en {tb_dir}")
        plt.close(fig)
        return data

    ax.set_title(f"Métricas F1 de validación — {exp['label']}")
    ax.set_xlabel('Época')
    ax.set_ylabel('F1-Score')
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()

    save_fig(fig, f"f1_{exp['name']}")
    plt.close(fig)
    return data


def plot_combined(all_data):
    """
    Plot combinado: 3 subplots (uno por métrica F1), cada uno con las 3 experimentos.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle('Comparativa de métricas F1 de validación', fontsize=13, y=1.02)

    for ax, metric in zip(axes, F1_METRICS):
        has_any = False
        for exp, exp_data in zip(EXPERIMENTS, all_data):
            steps, values = exp_data.get(metric['tag'], (None, None))
            if steps is None:
                continue
            ax.plot(steps, values, color=exp['color'],
                    linewidth=1.8, label=exp['label'])
            has_any = True

        ax.set_title(metric['label'])
        ax.set_xlabel('Época')
        ax.set_ylabel('F1-Score')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if has_any:
            ax.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, 'f1_all_experiments')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Genera plots de métricas F1 desde eventos TensorBoard")
    parser.add_argument(
        '--exp', type=str, default=None,
        help='Nombre del experimento. Sin --exp genera todos.'
    )
    args = parser.parse_args()

    targets = [e for e in EXPERIMENTS if args.exp is None or e['name'] == args.exp]

    if not targets:
        print(f"Experimento desconocido: '{args.exp}'")
        print(f"Opciones: {[e['name'] for e in EXPERIMENTS]}")
        return

    base = _DRIVE_BASE if _IN_COLAB else _LOCAL_BASE
    print(f"Leyendo eventos desde: {base}")
    print(f"Guardando plots en:    {PLOTS_DIR}/\n")

    # Cargar datos de todos los experimentos (para el combinado)
    all_data = []
    for exp in EXPERIMENTS:
        if exp in targets:
            print(f"-> {exp['name']}")
            data = plot_single(exp, base)
        else:
            # Solo cargar para el combinado, sin generar plot individual
            tb_dir = os.path.join(base, exp['name'])
            data   = {m['tag']: load_scalars(tb_dir, m['tag']) for m in F1_METRICS}
        all_data.append(data)

    print(f"\n-> Combinado")
    plot_combined(all_data)
    print("\nListo.")


if __name__ == '__main__':
    main()
