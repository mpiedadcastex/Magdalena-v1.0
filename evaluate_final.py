"""
evaluate_final.py — Evaluación final Magdalena v1.0 sobre el test set de MAESTRO.

Evalúa los tres modelos entrenados sobre una muestra aleatoria de 30 piezas
del test set (seed=42), usando temperature sampling (T=0.8).

Uso habitual (Colab con GPU, ~1-2 h):
    python evaluate_final.py

Verificación rápida sin generación (local, CPU):
    python evaluate_final.py --dry_run

Rutas para Colab: ajustar CSV_PATH y ROOT_DIR en la sección de configuración.

Salida:
    results/v1_baseline_test_results.csv
    results/exp01_test_results.csv
    results/exp02_test_results.csv
    results/predictions/{exp_name}/*.mid   (MIDIs generados)
    results/summary_test_results.txt
    results/best_transcription_comparison.png
"""

import os
import sys
import argparse
import random
import traceback
import tempfile

import numpy as np
import pandas as pd
import torch
import pretty_midi
import mir_eval
import mir_eval.transcription
import mir_eval.transcription_velocity
import matplotlib
matplotlib.use('Agg')   # sin pantalla (compatible con Colab y servidores)
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data.audio_proc import AudioProcessor
from src.data.midi_proc import MidiProcessor
from src.models.transformer import PianoTranscriptionModel
from utils import get_model_config

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — detección automática de entorno (Colab vs local)
# ═══════════════════════════════════════════════════════════════════════════════

_IN_COLAB = os.path.isdir('/content/drive')

if _IN_COLAB:
    CSV_PATH   = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv'
    ROOT_DIR   = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0'
    _CKPT_BASE = '/content/drive/MyDrive/TFG_Project/MPCS/checkpoints'
    # En Drive los checkpoints no tienen el sufijo de ranking local (-001, -002…)
    _CKPT = {
        'v1_baseline':        f'{_CKPT_BASE}/v1_baseline/model/model_epoch_98.pth',
        'exp01_reduced_data': f'{_CKPT_BASE}/exp01_reduced_data/model/model_epoch_514.pth',
        'exp02_short_clips':  f'{_CKPT_BASE}/exp02_short_clips/model/model_epoch_57.pth',
    }
else:
    CSV_PATH   = 'data/maestro-v3.0.0_metadata.csv'
    ROOT_DIR   = 'data/raw/maestro-v3.0.0/maestro-v3.0.0'
    # Localmente los archivos descargados incluyen sufijo de ranking
    _CKPT = {
        'v1_baseline':        'checkpoints/v1_baseline/model/model_epoch_98-001.pth',
        'exp01_reduced_data': 'checkpoints/exp01_reduced_data/model/model_epoch_514.pth',
        'exp02_short_clips':  'checkpoints/exp02_short_clips/model/model_epoch_57-001.pth',
    }

RESULTS_DIR = 'results'

# Checkpoints: sufijo -001 en nombres locales indica mejor val_loss según ranking.
# exp01 no tiene ranking; se usa la época más reciente disponible (514).
EXPERIMENTS = [
    {
        'name':             'v1_baseline',
        'label':            'v1_baseline',
        'checkpoint':       _CKPT['v1_baseline'],
        'epoch':            98,
        'max_audio_frames': 4096,   # 81.92 s (hop=320, fs=16000)
        'csv_out':          'results/v1_baseline_test_results.csv',
        'val_loss_note':    'mejor checkpoint (epoca 98)',
    },
    {
        'name':             'exp01_reduced_data',
        'label':            'exp01',
        'checkpoint':       _CKPT['exp01_reduced_data'],
        'epoch':            514,
        'max_audio_frames': 4096,
        'csv_out':          'results/exp01_test_results.csv',
        'val_loss_note':    'ultima epoca disponible (sin log de val_loss)',
    },
    {
        'name':             'exp02_short_clips',
        'label':            'exp02',
        'checkpoint':       _CKPT['exp02_short_clips'],
        'epoch':            57,
        'max_audio_frames': 2048,   # 40.96 s — igual que durante el entrenamiento
        'csv_out':          'results/exp02_test_results.csv',
        'val_loss_note':    'mejor checkpoint (epoca 57)',
    },
]

# Parámetros de evaluación
DEVICE             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SAMPLES          = 30         # piezas a evaluar (muestra aleatoria con seed fija)
RANDOM_SEED        = 42
TEMPERATURE        = 0.8
MAX_GEN_TOKENS     = 3000       # límite de tokens generados por pieza
FPS                = 50         # frames/s  (hop_length=320 / sample_rate=16000)
ONSET_TOLERANCE    = 0.05       # 50 ms
OFFSET_RATIO       = 0.2
VELOCITY_TOLERANCE = 0.1

# Arquitectura (idéntica en los tres experimentos)
EMBED_DIM          = 256
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
NHEAD              = 4


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def midi_to_intervals(midi_path, max_time=None):
    """
    Lee un archivo MIDI y devuelve (intervals, pitches, velocities).
    max_time: si se indica, se descartan las notas que empiezan después de ese tiempo.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"  [WARN] No se pudo leer {os.path.basename(midi_path)}: {e}")
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    intervals, pitches, velocities = [], [], []
    for inst in pm.instruments:
        for note in inst.notes:
            if max_time is not None and note.start >= max_time:
                continue
            end = min(note.end, max_time) if max_time is not None else note.end
            intervals.append([note.start, end])
            pitches.append(float(note.pitch))
            velocities.append(float(note.velocity))

    if not intervals:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    return (np.array(intervals, dtype=float),
            np.array(pitches,   dtype=float),
            np.array(velocities, dtype=float))


def compute_metrics(ref_int, ref_p, ref_v, est_int, est_p, est_v):
    """Devuelve (f1_onset, f1_onset_offset, f1_onset_velocity)."""
    if est_int.shape[0] == 0 or ref_int.shape[0] == 0:
        return 0.0, 0.0, 0.0

    _, _, on_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_p, est_int, est_p,
        onset_tolerance=ONSET_TOLERANCE, offset_ratio=None)

    _, _, on_off_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_p, est_int, est_p,
        onset_tolerance=ONSET_TOLERANCE, offset_ratio=OFFSET_RATIO)

    _, _, vel_f1, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_int, ref_p, ref_v, est_int, est_p, est_v,
        onset_tolerance=ONSET_TOLERANCE, offset_ratio=OFFSET_RATIO,
        velocity_tolerance=VELOCITY_TOLERANCE)

    return float(on_f1), float(on_off_f1), float(vel_f1)


def predict_sampling(model, audio_tensor, mp):
    """
    Generación autorregressiva con temperature sampling (T=0.8).
    El encoder se ejecuta UNA sola vez y su salida (memory) se reutiliza
    en cada paso del decoder — crítico para velocidad de inferencia.
    """
    model.eval()
    device    = audio_tensor.device
    generated = torch.tensor([[mp.token_sos]], dtype=torch.long, device=device)

    with torch.no_grad():
        # Codificar el audio una sola vez: (1, T_audio, d_model)
        memory = model.encoder(audio_tensor)

        for _ in range(MAX_GEN_TOKENS):
            logits     = model.decoder(tgt=generated, memory=memory,
                                       tgt_padding_mask=None)
            next_logit = logits[:, -1, :] / TEMPERATURE
            probs      = torch.softmax(next_logit, dim=-1)
            next_tok   = torch.multinomial(probs, num_samples=1)

            if next_tok.item() == mp.token_eos:
                break

            generated = torch.cat([generated, next_tok], dim=1)

    return generated.squeeze().cpu().numpy()


def load_model(exp):
    """Instancia y carga el modelo con el checkpoint del experimento."""
    cfg   = get_model_config()
    model = PianoTranscriptionModel(
        midi_processor=MidiProcessor(),
        encoder_cfg=cfg,
        embed_dim=EMBED_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        nhead=NHEAD,
    ).to(DEVICE)
    state = torch.load(exp['checkpoint'], map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"  Checkpoint: {exp['checkpoint']}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN DE UN EXPERIMENTO
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_experiment(exp, test_rows, ap, mp, dry_run=False):
    """
    Evalúa un experimento sobre test_rows.
    Guarda los MIDIs predichos en results/predictions/{exp_name}/.
    Devuelve un DataFrame con resultados por pieza.
    """
    max_audio_frames = exp['max_audio_frames']
    max_time         = max_audio_frames / FPS

    pred_dir = os.path.join(RESULTS_DIR, 'predictions', exp['name'])
    os.makedirs(pred_dir, exist_ok=True)

    model = load_model(exp)
    rows  = []

    for i, (_, row) in enumerate(tqdm(test_rows.iterrows(),
                                      total=len(test_rows),
                                      desc=f"  {exp['name']}")):
        audio_path = os.path.join(ROOT_DIR, row['audio_filename'])
        midi_path  = os.path.join(ROOT_DIR, row['midi_filename'])
        filename   = row['audio_filename']

        try:
            # Espectrograma
            spec = ap.compute_spectogram(audio_path)          # (229, T)
            spec = torch.from_numpy(spec).float()
            if spec.shape[1] > max_audio_frames:
                spec = spec[:, :max_audio_frames]
            audio_tensor = spec.unsqueeze(0).to(DEVICE)       # (1, 229, T_clip)

            # Ground truth
            ref_int, ref_p, ref_v = midi_to_intervals(midi_path, max_time=max_time)
            n_notes_gt = int(len(ref_p))

            if dry_run:
                rows.append({'filename': filename, 'n_notes_gt': n_notes_gt,
                             'n_notes_pred': -1, 'f1_onset': -1.0,
                             'f1_onset_offset': -1.0, 'f1_onset_velocity': -1.0,
                             'pred_midi_path': ''})
                break   # una pieza en dry_run

            # Generación
            pred_tokens  = predict_sampling(model, audio_tensor, mp)
            pred_path    = os.path.join(pred_dir, f"pred_{i:03d}.mid")
            mp.decode_midi(pred_tokens, output_path=pred_path)

            # Métricas
            est_int, est_p, est_v = midi_to_intervals(pred_path, max_time=max_time)
            n_notes_pred = int(len(est_p))

            on_f1, on_off_f1, vel_f1 = compute_metrics(
                ref_int, ref_p, ref_v,
                est_int, est_p, est_v)

            rows.append({
                'filename':          filename,
                'n_notes_gt':        n_notes_gt,
                'n_notes_pred':      n_notes_pred,
                'f1_onset':          round(on_f1,     4),
                'f1_onset_offset':   round(on_off_f1, 4),
                'f1_onset_velocity': round(vel_f1,    4),
                'pred_midi_path':    pred_path,
            })

            print(f"    [{i+1:02d}] onset={on_f1:.3f}  o+off={on_off_f1:.3f}"
                  f"  vel={vel_f1:.3f}  gt={n_notes_gt}n  pred={n_notes_pred}n"
                  f"  {os.path.basename(filename)[:35]}")

        except Exception:
            traceback.print_exc()
            rows.append({'filename': filename, 'n_notes_gt': -1,
                         'n_notes_pred': -1, 'f1_onset': 0.0,
                         'f1_onset_offset': 0.0, 'f1_onset_velocity': 0.0,
                         'pred_midi_path': ''})

    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    # Guardar CSV con columnas requeridas (sin pred_midi_path)
    df[['filename', 'n_notes_gt', 'n_notes_pred',
        'f1_onset', 'f1_onset_offset', 'f1_onset_velocity']].to_csv(
        exp['csv_out'], index=False)
    print(f"  CSV guardado: {exp['csv_out']}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════════

def _agg(df, col):
    valid = df.loc[df[col] >= 0.0, col]
    return dict(mean=valid.mean(), std=valid.std(),
                median=valid.median(), min=valid.min(), max=valid.max())


def _fmtrow(label, s):
    return (f"  {label:<22}| {s['mean']:6.3f}  | {s['std']:5.3f}"
            f" | {s['median']:7.3f}   | {s['min']:5.3f} | {s['max']:5.3f}")


def write_summary(all_results):
    lines = []
    sep   = "  ══════════════════════════════════════════════════════"
    lines += [sep, "  RESUMEN DE EVALUACIÓN — TEST SET MAESTRO",
              f"  Muestras por modelo: {N_SAMPLES}  (seed={RANDOM_SEED})", sep, ""]

    comparative = {}

    for exp, df in zip(EXPERIMENTS, all_results):
        valid  = df[df['f1_onset'] >= 0.0]
        n_eval = len(valid)
        s_on   = _agg(df, 'f1_onset')
        s_oof  = _agg(df, 'f1_onset_offset')
        s_vel  = _agg(df, 'f1_onset_velocity')

        above  = (valid['f1_onset'] > 0.5).sum()
        pct    = 100 * above / n_eval if n_eval > 0 else 0

        gt_mean   = valid['n_notes_gt'].mean()   if 'n_notes_gt'   in df else float('nan')
        pred_mean = valid['n_notes_pred'].mean() if 'n_notes_pred' in df else float('nan')

        comparative[exp['label']] = (s_on['mean'], s_oof['mean'], s_vel['mean'])

        lines += [
            f"  Modelo: {exp['name']}",
            f"  Checkpoint: época {exp['epoch']}  ({exp['val_loss_note']})",
            f"  Piezas evaluadas: {n_eval}",
            "",
            f"  {'Métrica':<22}| {'Media':^6}  | {'Std':^5} | {'Mediana':^7}   | {'Min':^5} | {'Max':^5}",
            f"  {'-'*22}|{'-'*8}|{'-'*7}|{'-'*9}|{'-'*7}|{'-'*6}",
            _fmtrow("F1 Onset",            s_on),
            _fmtrow("F1 Onset + Offset",   s_oof),
            _fmtrow("F1 Onset + Velocity", s_vel),
            "",
            f"  Notas GT (media):    {gt_mean:.0f}",
            f"  Notas pred. (media): {pred_mean:.0f}",
            f"  Piezas F1>0.5:       {above}/{n_eval} ({pct:.1f}%)",
            "",
        ]

    lines += [sep, "  TABLA COMPARATIVA", sep, ""]
    labels = [e['label'] for e in EXPERIMENTS]
    header = f"  {'Métrica':<22}| " + " | ".join(f"{l:^11}" for l in labels)
    lines.append(header)
    lines.append(f"  {'-'*22}|" + "|".join(["-"*13]*len(labels)))

    for metric_name, idx in [("F1 Onset (media)", 0),
                              ("F1 O+Off (media)", 1),
                              ("F1 O+Vel (media)", 2)]:
        vals = " | ".join(f"{comparative[l][idx]:^11.3f}" for l in labels)
        lines.append(f"  {metric_name:<22}| {vals}")

    lines.append("")

    out_path = os.path.join(RESULTS_DIR, 'summary_test_results.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    print(f"Resumen guardado en {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS CUALITATIVO
# ═══════════════════════════════════════════════════════════════════════════════

def qualitative_analysis(df, exp_name, metadata):
    valid = df[df['f1_onset'] >= 0.0].copy()
    if len(valid) < 3:
        print("  Insuficientes piezas para análisis cualitativo.")
        return

    valid = valid.merge(
        metadata[['audio_filename', 'canonical_composer', 'duration']],
        left_on='filename', right_on='audio_filename', how='left')

    print(f"\n── Análisis cualitativo: {exp_name} ──")
    for label, subset in [("3 mejores", valid.nlargest(3,  'f1_onset')),
                           ("3 peores",  valid.nsmallest(3, 'f1_onset'))]:
        print(f"\n{label} transcripciones (F1 onset):")
        for _, r in subset.iterrows():
            print(f"  F1={r['f1_onset']:.3f}  "
                  f"{str(r.get('canonical_composer','?')):<30}  "
                  f"dur={r.get('duration', 0):.0f}s  "
                  f"{os.path.basename(str(r['filename']))}")


# ═══════════════════════════════════════════════════════════════════════════════
# PIANO ROLL
# ═══════════════════════════════════════════════════════════════════════════════

def piano_roll_comparison(gt_midi_path, pred_midi_path, out_path, max_time, title=""):
    """Genera y guarda un piano roll comparativo GT vs. predicción."""
    def get_roll(path):
        pm   = pretty_midi.PrettyMIDI(path)
        roll = pm.get_piano_roll(fs=FPS)
        n_frames = int(max_time * FPS)
        return roll[:, :n_frames] if roll.shape[1] >= n_frames else roll

    gt_roll   = get_roll(gt_midi_path)
    pred_roll = get_roll(pred_midi_path)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].imshow(gt_roll,   aspect='auto', origin='lower', cmap='Blues',   vmin=0, vmax=127)
    axes[0].set_title(f"Ground truth — {title}", fontsize=10)
    axes[0].set_ylabel("Pitch (MIDI)")
    axes[1].imshow(pred_roll, aspect='auto', origin='lower', cmap='Oranges', vmin=0, vmax=127)
    axes[1].set_title("Predicción Magdalena v1.0", fontsize=10)
    axes[1].set_ylabel("Pitch (MIDI)")
    axes[1].set_xlabel(f"Trama (fps={FPS})")
    plt.suptitle(f"Mejor transcripción — {exp_name}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Piano roll guardado en {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluación final Magdalena v1.0")
    parser.add_argument('--dry_run', action='store_true',
                        help='Carga 1 pieza sin generación para verificar imports y pesos')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"CSV:    {CSV_PATH}")
    print(f"Root:   {ROOT_DIR}\n")

    # Muestra aleatoria del test split (seed fija para reproducibilidad)
    metadata  = pd.read_csv(CSV_PATH)
    test_df   = metadata[metadata['split'] == 'test'].reset_index(drop=True)
    n         = 1 if args.dry_run else min(N_SAMPLES, len(test_df))
    rng       = random.Random(RANDOM_SEED)
    indices   = rng.sample(range(len(test_df)), n)
    test_rows = test_df.iloc[indices].reset_index(drop=True)

    print(f"Test split total: {len(test_df)} piezas")
    print(f"Piezas seleccionadas: {len(test_rows)}"
          + (" (dry run)" if args.dry_run else f" (seed={RANDOM_SEED})"))

    ap = AudioProcessor()
    mp = MidiProcessor()

    all_results   = []
    best_model_df = None
    best_model_i  = -1
    best_mean_f1  = -1.0

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f" Evaluando: {exp['name']}  |  epoca {exp['epoch']}")
        print(f"{'='*60}")

        df = evaluate_experiment(exp, test_rows, ap, mp, dry_run=args.dry_run)
        all_results.append(df)

        if not args.dry_run:
            mean_f1 = df.loc[df['f1_onset'] >= 0, 'f1_onset'].mean()
            if mean_f1 > best_mean_f1:
                best_mean_f1  = mean_f1
                best_model_i  = i
                best_model_df = df

    if args.dry_run:
        print("\n[DRY RUN OK] Imports y carga de pesos correctos.")
        print("Ejecuta sin --dry_run para la evaluación completa (requiere GPU).")
        return

    # Resumen completo
    write_summary(all_results)

    # Análisis cualitativo del mejor modelo
    best_exp = EXPERIMENTS[best_model_i]
    qualitative_analysis(best_model_df, best_exp['name'], metadata)

    # Piano roll de la mejor pieza del mejor modelo
    valid_best = best_model_df[best_model_df['f1_onset'] >= 0].copy()
    if not valid_best.empty and 'pred_midi_path' in valid_best.columns:
        top_row     = valid_best.loc[valid_best['f1_onset'].idxmax()]
        gt_path     = os.path.join(ROOT_DIR, top_row['filename'])
        pred_path   = top_row['pred_midi_path']
        piano_path  = os.path.join(RESULTS_DIR, 'best_transcription_comparison.png')

        if os.path.exists(gt_path) and os.path.exists(pred_path):
            exp_name = best_exp['name']
            max_time = best_exp['max_audio_frames'] / FPS
            piano_roll_comparison(gt_path, pred_path, piano_path,
                                  max_time=max_time,
                                  title=os.path.basename(top_row['filename']))


if __name__ == '__main__':
    main()
