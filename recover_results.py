"""
recover_results.py — Reconstruye los archivos de resultados de la evaluación final
a partir de los valores extraídos del output de consola de evaluate_final.py.

No requiere GPU ni reejecutar el modelo.
Uso: python recover_results.py
"""

import os
import random
import numpy as np
import pandas as pd

# ─── Rutas ────────────────────────────────────────────────────────────────────
_IN_COLAB = os.path.isdir('/content/drive')
if _IN_COLAB:
    CSV_PATH = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv'
else:
    CSV_PATH = 'data/maestro-v3.0.0_metadata.csv'

RESULTS_DIR  = 'results'
RANDOM_SEED  = 42
N_SAMPLES    = 30

# ─── Métricas extraídas del output de consola ─────────────────────────────────
# Orden idéntico al de evaluación (seed=42, muestra reproducible del test split)
# Formato: (f1_onset, f1_onset_offset, f1_onset_velocity, n_notes_gt, n_notes_pred)

RAW = {
    'v1_baseline': [
        (0.070, 0.022, 0.018, 1361,  636),
        (0.071, 0.016, 0.007, 1062,  637),
        (0.066, 0.010, 0.010,  632,  613),
        (0.081, 0.043, 0.019, 1126,  642),
        (0.064, 0.009, 0.000,  561,  593),
        (0.058, 0.007, 0.007,  900,  610),
        (0.041, 0.019, 0.007,  528,  637),
        (0.085, 0.038, 0.024,  874,  616),
        (0.059, 0.010, 0.007,  856,  606),
        (0.125, 0.052, 0.026,  696,  600),
        (0.042, 0.016, 0.012,  833,  641),
        (0.139, 0.064, 0.047,  747,  620),
        (0.078, 0.033, 0.021, 1251,  637),
        (0.084, 0.046, 0.021, 1346,  626),
        (0.116, 0.048, 0.034,  748,  614),
        (0.063, 0.021, 0.015,  814,  698),
        (0.126, 0.053, 0.031,  831,  608),
        (0.071, 0.032, 0.009,  949,  626),
        (0.104, 0.041, 0.022, 1633,  645),
        (0.095, 0.027, 0.014,  793,  615),
        (0.068, 0.028, 0.013,  717,  642),
        (0.083, 0.034, 0.010, 1148,  630),
        (0.054, 0.003, 0.003,  420,  315),
        (0.102, 0.044, 0.032,  770,  626),
        (0.059, 0.017, 0.010,  920,  643),
        (0.066, 0.011, 0.011,  501,  616),
        (0.064, 0.008, 0.004,  355,  643),
        (0.107, 0.035, 0.020,  991,  602),
        (0.060, 0.014, 0.002,  500,  638),
        (0.107, 0.054, 0.032, 1455,  667),
    ],
    'exp01_reduced_data': [
        (0.053, 0.013, 0.005, 1361,  549),
        (0.076, 0.008, 0.002, 1062,  657),
        (0.065, 0.003, 0.003,  632,  625),
        (0.071, 0.020, 0.008, 1126,  574),
        (0.063, 0.014, 0.007,  561,  580),
        (0.083, 0.019, 0.015,  900,  612),
        (0.032, 0.008, 0.002,  528,  665),
        (0.055, 0.010, 0.008,  874,  574),
        (0.076, 0.007, 0.001,  856,  677),
        (0.063, 0.007, 0.002,  696,  479),
        (0.067, 0.010, 0.004,  833,  534),
        (0.092, 0.029, 0.016,  747,  725),
        (0.115, 0.027, 0.009, 1251,  645),
        (0.068, 0.019, 0.009, 1346,  754),
        (0.062, 0.010, 0.006,  748,  633),
        (0.065, 0.009, 0.003,  814,  667),
        (0.088, 0.028, 0.022,  831,  605),
        (0.035, 0.004, 0.002,  949, 1048),
        (0.088, 0.023, 0.014, 1633,  612),
        (0.067, 0.011, 0.007,  793,  678),
        (0.052, 0.008, 0.004,  717,  749),
        (0.063, 0.012, 0.005, 1148,  682),
        (0.046, 0.011, 0.007,  420,  459),
        (0.061, 0.007, 0.004,  770,  704),
        (0.082, 0.019, 0.015,  920,  552),
        (0.050, 0.012, 0.002,  501,  534),
        (0.043, 0.008, 0.006,  355,  666),
        (0.073, 0.011, 0.004,  991,  480),
        (0.048, 0.002, 0.002,  500,  369),
        (0.058, 0.017, 0.011, 1455,  637),
    ],
    'exp02_short_clips': [
        (0.134, 0.046, 0.028,  635,  637),
        (0.092, 0.012, 0.008,  514,  512),
        (0.061, 0.012, 0.006,  262,  399),
        (0.078, 0.014, 0.008,  577,  709),
        (0.045, 0.005, 0.005,  251,  555),
        (0.101, 0.022, 0.018,  443,  546),
        (0.055, 0.016, 0.016,  274,  343),
        (0.110, 0.027, 0.018,  406,  488),
        (0.088, 0.029, 0.009,  452,  644),
        (0.146, 0.050, 0.034,  353,  535),
        (0.080, 0.013, 0.004,  468,  630),
        (0.185, 0.085, 0.067,  468,  637),
        (0.188, 0.086, 0.033,  592,  624),
        (0.096, 0.031, 0.020,  836,  659),
        (0.149, 0.062, 0.029,  385,  648),
        (0.078, 0.008, 0.008,  412,  643),
        (0.122, 0.039, 0.018,  450,  632),
        (0.066, 0.016, 0.002,  453,  695),
        (0.170, 0.045, 0.028,  926,  506),
        (0.135, 0.038, 0.027,  407,  641),
        (0.072, 0.014, 0.005,  292,  541),
        (0.097, 0.031, 0.005,  610,  663),
        (0.063, 0.008, 0.008,  222,  287),
        (0.123, 0.035, 0.029,  473,  435),
        (0.118, 0.010, 0.006,  458,  591),
        (0.052, 0.007, 0.007,  242,  297),
        (0.028, 0.004, 0.004,  157,  336),
        (0.113, 0.041, 0.023,  508,  462),
        (0.088, 0.006, 0.006,  236,  402),
        (0.152, 0.055, 0.030,  753,  586),
    ],
}

EXPERIMENTS = [
    {'name': 'v1_baseline',        'epoch': 98,  'label': 'v1_baseline',
     'val_loss_note': 'mejor checkpoint (epoca 98)',
     'csv_out': 'results/v1_baseline_test_results.csv'},
    {'name': 'exp01_reduced_data', 'epoch': 514, 'label': 'exp01',
     'val_loss_note': 'ultima epoca disponible (sin log de val_loss)',
     'csv_out': 'results/exp01_test_results.csv'},
    {'name': 'exp02_short_clips',  'epoch': 57,  'label': 'exp02',
     'val_loss_note': 'mejor checkpoint (epoca 57)',
     'csv_out': 'results/exp02_test_results.csv'},
]


def reproduce_test_rows(csv_path, n=30, seed=42):
    """Reproduce exactamente la muestra usada durante la evaluación."""
    meta    = pd.read_csv(csv_path)
    test_df = meta[meta['split'] == 'test'].reset_index(drop=True)
    rng     = random.Random(seed)
    indices = rng.sample(range(len(test_df)), n)
    return test_df.iloc[indices].reset_index(drop=True)


def build_df(exp_name, test_rows):
    data   = RAW[exp_name]
    rows   = []
    for i, row in test_rows.iterrows():
        on, oof, vel, gt, pred = data[i]
        rows.append({
            'filename':          row['audio_filename'],
            'n_notes_gt':        gt,
            'n_notes_pred':      pred,
            'f1_onset':          on,
            'f1_onset_offset':   oof,
            'f1_onset_velocity': vel,
        })
    return pd.DataFrame(rows)


def _agg(df, col):
    s = df[col]
    return dict(mean=s.mean(), std=s.std(), median=s.median(),
                min=s.min(), max=s.max())


def _fmtrow(label, s):
    return (f"  {label:<22}| {s['mean']:6.3f}  | {s['std']:5.3f}"
            f" | {s['median']:7.3f}   | {s['min']:5.3f} | {s['max']:5.3f}")


def write_summary(all_dfs):
    lines = []
    sep   = "  ══════════════════════════════════════════════════════"
    lines += [sep, "  RESUMEN DE EVALUACIÓN — TEST SET MAESTRO",
              f"  Muestras por modelo: {N_SAMPLES}  (seed={RANDOM_SEED})", sep, ""]

    comparative = {}
    for exp, df in zip(EXPERIMENTS, all_dfs):
        s_on  = _agg(df, 'f1_onset')
        s_oof = _agg(df, 'f1_onset_offset')
        s_vel = _agg(df, 'f1_onset_velocity')
        above = (df['f1_onset'] > 0.5).sum()
        pct   = 100 * above / len(df)
        comparative[exp['label']] = (s_on['mean'], s_oof['mean'], s_vel['mean'])

        lines += [
            f"  Modelo: {exp['name']}",
            f"  Checkpoint: epoca {exp['epoch']}  ({exp['val_loss_note']})",
            f"  Piezas evaluadas: {len(df)}",
            "",
            f"  {'Metrica':<22}| {'Media':^6}  | {'Std':^5} | {'Mediana':^7}   | {'Min':^5} | {'Max':^5}",
            f"  {'-'*22}|{'-'*8}|{'-'*7}|{'-'*9}|{'-'*7}|{'-'*6}",
            _fmtrow("F1 Onset",            s_on),
            _fmtrow("F1 Onset + Offset",   s_oof),
            _fmtrow("F1 Onset + Velocity", s_vel),
            "",
            f"  Notas GT (media):    {df['n_notes_gt'].mean():.0f}",
            f"  Notas pred. (media): {df['n_notes_pred'].mean():.0f}",
            f"  Piezas F1>0.5:       {above}/{len(df)} ({pct:.1f}%)",
            "",
        ]

    labels = [e['label'] for e in EXPERIMENTS]
    lines += [sep, "  TABLA COMPARATIVA", sep, ""]
    lines.append(f"  {'Metrica':<22}| " + " | ".join(f"{l:^11}" for l in labels))
    lines.append(f"  {'-'*22}|" + "|".join(["-"*13]*len(labels)))
    for metric_name, idx in [("F1 Onset (media)", 0),
                              ("F1 O+Off (media)", 1),
                              ("F1 O+Vel (media)", 2)]:
        vals = " | ".join(f"{comparative[l][idx]:^11.3f}" for l in labels)
        lines.append(f"  {metric_name:<22}| {vals}")
    lines.append("")

    out = os.path.join(RESULTS_DIR, 'summary_test_results.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f"Resumen guardado en {out}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Leyendo metadata desde: {CSV_PATH}")
    test_rows = reproduce_test_rows(CSV_PATH, N_SAMPLES, RANDOM_SEED)
    print(f"Filas reproducidas: {len(test_rows)}\n")

    all_dfs = []
    for exp in EXPERIMENTS:
        df = build_df(exp['name'], test_rows)
        df.to_csv(exp['csv_out'], index=False)
        print(f"CSV guardado: {exp['csv_out']}")
        all_dfs.append(df)

    write_summary(all_dfs)

    # Análisis cualitativo: mejores y peores de exp02 (mejor modelo)
    meta   = pd.read_csv(CSV_PATH)
    df_exp02 = all_dfs[2].merge(
        meta[['audio_filename', 'canonical_composer', 'duration']],
        left_on='filename', right_on='audio_filename', how='left')

    print("\n-- Analisis cualitativo: exp02_short_clips --")
    for label, subset in [("3 mejores", df_exp02.nlargest(3,  'f1_onset')),
                           ("3 peores",  df_exp02.nsmallest(3, 'f1_onset'))]:
        print(f"\n{label} transcripciones (F1 onset):")
        for _, r in subset.iterrows():
            print(f"  F1={r['f1_onset']:.3f}  "
                  f"{str(r.get('canonical_composer','?')):<30}  "
                  f"dur={r.get('duration', 0):.0f}s  "
                  f"{os.path.basename(str(r['filename']))}")


if __name__ == '__main__':
    main()
