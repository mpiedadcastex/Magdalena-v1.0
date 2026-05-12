"""
Genera un informe académico en Markdown con los resultados de un experimento.

Uso:
    python scripts/report_experiment.py --exp exp01_reduced_data
"""
import argparse
import importlib
import os
import re
from datetime import datetime
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description="Genera el informe de resultados de un experimento.")
    parser.add_argument(
        '--exp', type=str, required=True,
        help='Nombre del experimento (ej: exp01_reduced_data).'
    )
    return parser.parse_args()


def load_experiment_config(exp_name):
    """
    Importa dinámicamente el config.py del experimento indicado.

    Args:
        exp_name (str): Nombre de la carpeta del experimento.

    Returns:
        SimpleNamespace con todos los atributos del config.
    """
    module = importlib.import_module(f"experiments.{exp_name}.config")
    return SimpleNamespace(**{
        k: getattr(module, k)
        for k in dir(module)
        if not k.startswith('_')
    })


def load_baseline_config():
    """
    Devuelve los hiperparámetros de v1_baseline para la tabla comparativa.

    Returns:
        SimpleNamespace con los valores de producción.
    """
    return SimpleNamespace(
        MAX_SAMPLES_TRAIN="Todas (~962)",
        MAX_SAMPLES_VAL="Todas",
        MAX_AUDIO_FRAMES=4096,
        MAX_MIDI_TOKENS=1500,
        BATCH_SIZE=2,
        GRAD_ACCUMULATION_STEPS=8,
        LEARNING_RATE=1e-4,
        EPOCHS=80,
        EMBED_DIM=256,
        NUM_ENCODER_LAYERS=4,
        NUM_DECODER_LAYERS=4,
        NHEAD=4,
    )


def parse_loss_log(log_path):
    """
    Parsea el training_loss_log.txt y devuelve los registros deduplicados por época.

    Args:
        log_path (str): Ruta completa al archivo de log.

    Returns:
        list[dict]: Lista ordenada por época con claves epoch, loss, timestamp.
                    Lista vacía si el archivo no existe.
    """
    if not os.path.exists(log_path):
        return []

    pattern = re.compile(
        r'\|\s*([\d\-: ]+?)\s*\|\s*Epoch:\s*(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|'
    )

    entries = {}  # epoch -> dict, para deduplicar entradas duplicadas del log
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(2))
                if epoch not in entries:
                    entries[epoch] = {
                        'epoch': epoch,
                        'loss': float(match.group(3)),
                        'timestamp': datetime.strptime(match.group(1).strip(), "%Y-%m-%d %H:%M:%S")
                    }

    return sorted(entries.values(), key=lambda x: x['epoch'])


def compute_epoch_times(records):
    """
    Añade duration_min a cada registro calculando diferencias entre timestamps consecutivos.
    El último epoch no puede calcularse y se marca como None.

    Args:
        records (list[dict]): Registros ordenados por época.

    Returns:
        list[dict]: Los mismos registros con duration_min añadido.
    """
    for i in range(len(records) - 1):
        delta = records[i + 1]['timestamp'] - records[i]['timestamp']
        records[i]['duration_min'] = delta.total_seconds() / 60
    if records:
        records[-1]['duration_min'] = None
    return records


def compute_stats(records):
    """
    Calcula estadísticas globales del entrenamiento.

    Args:
        records (list[dict]): Registros con loss y duration_min.

    Returns:
        dict con min_loss, min_epoch, avg_duration_min, total_time_h, epochs_run.
        None si no hay registros.
    """
    if not records:
        return None

    losses    = [r['loss'] for r in records]
    durations = [r['duration_min'] for r in records if r['duration_min'] is not None]
    min_loss  = min(losses)

    return {
        'min_loss':         min_loss,
        'min_epoch':        next(r['epoch'] for r in records if r['loss'] == min_loss),
        'avg_duration_min': sum(durations) / len(durations) if durations else None,
        'total_time_h':     sum(durations) / 60 if durations else None,
        'epochs_run':       len(records),
    }


def format_config_table(cfg, baseline):
    """
    Genera la tabla comparativa de hiperparámetros marcando con ← los valores modificados.

    Args:
        cfg (SimpleNamespace): Config del experimento.
        baseline (SimpleNamespace): Config de v1_baseline.

    Returns:
        str: Tabla en formato Markdown.
    """
    params = [
        ("MAX_SAMPLES_TRAIN",       "Muestras de entrenamiento"),
        ("MAX_SAMPLES_VAL",         "Muestras de validación"),
        ("MAX_AUDIO_FRAMES",        "Frames de audio máximos"),
        ("MAX_MIDI_TOKENS",         "Tokens MIDI máximos"),
        ("BATCH_SIZE",              "Batch size"),
        ("GRAD_ACCUMULATION_STEPS", "Gradient accumulation"),
        ("LEARNING_RATE",           "Learning rate"),
        ("EPOCHS",                  "Épocas"),
        ("EMBED_DIM",               "Dimensión de embedding"),
        ("NUM_ENCODER_LAYERS",      "Capas del encoder"),
        ("NUM_DECODER_LAYERS",      "Capas del decoder"),
        ("NHEAD",                   "Attention heads"),
    ]

    lines = [
        "| Parámetro | Este experimento | v1_baseline |",
        "|---|---|---|",
    ]
    for attr, label in params:
        exp_val  = getattr(cfg, attr, "—")
        base_val = getattr(baseline, attr, "—")
        changed  = str(exp_val) != str(base_val)
        marker   = " ←" if changed else ""
        lines.append(f"| {label} | **{exp_val}**{marker} | {base_val} |")

    return "\n".join(lines)


def format_results_table(records):
    """
    Genera la tabla de resultados por época.

    Args:
        records (list[dict]): Registros con epoch, loss, duration_min.

    Returns:
        str: Tabla en formato Markdown.
    """
    lines = [
        "| Época | Loss promedio | Tiempo (min) |",
        "|---|---|---|",
    ]
    for r in records:
        duration = f"{r['duration_min']:.1f}" if r['duration_min'] is not None else "—"
        lines.append(f"| {r['epoch']} | {r['loss']:.6f} | {duration} |")
    return "\n".join(lines)


def generate_report(cfg, records, baseline):
    """
    Construye el contenido completo del informe en Markdown.

    Args:
        cfg (SimpleNamespace): Config del experimento.
        records (list[dict]): Registros de entrenamiento parseados.
        baseline (SimpleNamespace): Config de v1_baseline.

    Returns:
        str: Contenido del informe.
    """
    today        = datetime.now().strftime("%Y-%m-%d")
    stats        = compute_stats(records)
    config_table = format_config_table(cfg, baseline)

    if records:
        results_section = format_results_table(records)
        stats_section = f"""\
### Estadísticas resumidas

| Métrica | Valor |
|---|---|
| Épocas ejecutadas | {stats['epochs_run']} |
| Loss mínimo | {stats['min_loss']:.6f} |
| Época del loss mínimo | {stats['min_epoch']} |
| Tiempo medio por época | {f"{stats['avg_duration_min']:.1f} min" if stats['avg_duration_min'] else "—"} |
| Tiempo total | {f"{stats['total_time_h']:.2f} h" if stats['total_time_h'] else "—"} |"""
    else:
        results_section = "*Sin datos todavía. Ejecuta el experimento y vuelve a generar el informe.*"
        stats_section   = ""

    return f"""\
# Experimento: `{cfg.EXP_NAME}`

**Fecha**: {today}
**\nObjetivo**: {cfg.EXP_DESCRIPTION}

---

## 1. Configuración

{config_table}

---

## 2. Resultados del entrenamiento

{results_section}

{stats_section}

---

## 3. Análisis

> *Completa esta sección con tus observaciones tras revisar los resultados.*

- **Convergencia**:
- **Velocidad respecto a v1_baseline**:
- **Calidad del aprendizaje**:
- **Conclusión**:

---

## 4. Próximos pasos

> *¿Qué ajustes o siguiente experimento sugieren estos resultados?*

"""


def main():
    args     = parse_args()
    cfg      = load_experiment_config(args.exp)
    baseline = load_baseline_config()

    log_path = os.path.join(cfg.DRIVE_LOG_PATH, "training_loss_log.txt")
    records  = parse_loss_log(log_path)
    records  = compute_epoch_times(records)

    if not records:
        print(f"AVISO: No se encontró log en {log_path}. Se generará el informe sin resultados.")

    report   = generate_report(cfg, records, baseline)

    output_path = os.path.join("experiments", args.exp, "results.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Informe generado: {output_path}")


if __name__ == "__main__":
    main()
