"""
Genera figures/audio_midi_alignment.png (300 dpi)

Pista: Frédéric Chopin — Ballade No. 1 in G Minor, Op. 23
Fuente: MAESTRO v3.0.0 (2004)
"""

import os
import numpy as np
import librosa
import pretty_midi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
os.makedirs('figures', exist_ok=True)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE = os.path.join('data', 'raw', 'maestro-v3.0.0', 'maestro-v3.0.0', '2004')
STEM = ('MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID'
        '--AUDIO_12_R1_2004_07_Track07_wav')
WAV  = os.path.join(BASE, STEM + '.wav')
MIDI = os.path.join(BASE, STEM + '.midi')

# ── Ventana de tiempo ─────────────────────────────────────────────────────────
SEG_START = 0.0
SEG_END   = 12.0

# ── Paleta ────────────────────────────────────────────────────────────────────
BG     = '#16213e'
PANEL  = '#1a1a2e'
TEXT   = 'white'
SPINE  = '#444466'
WAVE_C = '#4fc3f7'

BLACK_SEMITONES = {1, 3, 6, 8, 10}

def is_black(midi_note):
    return (midi_note % 12) in BLACK_SEMITONES

# ── Cargar audio ──────────────────────────────────────────────────────────────
print('Cargando audio...')
y, sr = librosa.load(WAV, sr=22050,
                     offset=SEG_START,
                     duration=SEG_END - SEG_START)
# t_audio alineado exactamente con SEG_START / SEG_END
t_audio = np.linspace(SEG_START, SEG_END, len(y))

# ── Cargar MIDI ───────────────────────────────────────────────────────────────
print('Cargando MIDI...')
pm = pretty_midi.PrettyMIDI(MIDI)
notes_in_seg = [
    note
    for inst in pm.instruments
    for note in inst.notes
    if note.start < SEG_END and note.end > SEG_START
]

# ── Figura ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7), facecolor=BG)
fig.patch.set_facecolor(BG)

# Coordenadas normalizadas [left, bottom, width, height]
# ax_wave y ax_roll comparten EXACTAMENTE el mismo left y width → alineación perfecta
LEFT_KB    = 0.030
WIDTH_KB   = 0.052
LEFT_MAIN  = LEFT_KB + WIDTH_KB        # 0.082
WIDTH_MAIN = 0.848                     # deja margen para colorbar (~1.4%)
CBAR_LEFT  = LEFT_MAIN + WIDTH_MAIN + 0.010
CBAR_WIDTH = 0.018

BOTTOM_ROLL = 0.09
HEIGHT_ROLL = 0.57
BOTTOM_WAVE = BOTTOM_ROLL + HEIGHT_ROLL   # sin hueco
HEIGHT_WAVE = 0.24

# Crear ejes con posiciones explícitas
ax_roll = fig.add_axes([LEFT_MAIN, BOTTOM_ROLL, WIDTH_MAIN, HEIGHT_ROLL])
ax_wave = fig.add_axes([LEFT_MAIN, BOTTOM_WAVE, WIDTH_MAIN, HEIGHT_WAVE])
ax_wave.sharex(ax_roll)                # garantía de sincronía de escala
ax_kb   = fig.add_axes([LEFT_KB,   BOTTOM_ROLL, WIDTH_KB,   HEIGHT_ROLL])

# ── Panel superior: Forma de onda ─────────────────────────────────────────────
ax_wave.set_facecolor(PANEL)
ax_wave.plot(t_audio, y, color=WAVE_C, linewidth=0.6, alpha=0.88)
ax_wave.set_xlim(SEG_START, SEG_END)
ax_wave.set_ylabel('Amplitud\n(normalizada)', color=TEXT, fontsize=9,
                   labelpad=6)
ax_wave.set_title(
    'Frédéric Chopin — Ballade No. 1 en Sol menor, Op. 23'
    '  (MAESTRO v3.0.0 · 2004)',
    color=TEXT, fontsize=11, pad=7, loc='left'
)
ax_wave.tick_params(colors=TEXT, labelsize=8,
                    bottom=False, labelbottom=False)
ax_wave.set_xticks([])
ax_wave.axhline(0, color=SPINE, linewidth=0.5, linestyle='--', alpha=0.6)
for sp in ax_wave.spines.values():
    sp.set_edgecolor(SPINE)
# Borde inferior = separador entre paneles
ax_wave.spines['bottom'].set_edgecolor('#556688')
ax_wave.spines['bottom'].set_linewidth(1.2)

# ── Teclado de piano ──────────────────────────────────────────────────────────
ax_kb.set_facecolor(PANEL)
ax_kb.set_xlim(0, 1)
ax_kb.set_ylim(0, 88)

for i in range(88):
    if not is_black(21 + i):
        ax_kb.add_patch(mpatches.Rectangle(
            (0, i), 1.0, 1.0,
            facecolor='#e8e8e8', edgecolor='#777',
            linewidth=0.3, zorder=1))
for i in range(88):
    if is_black(21 + i):
        ax_kb.add_patch(mpatches.Rectangle(
            (0, i), 0.52, 1.0,
            facecolor='#111122', edgecolor='#000',
            linewidth=0.2, zorder=2))
for i in range(88):
    midi_note = 21 + i
    if midi_note % 12 == 0:
        octave = midi_note // 12 - 1
        ax_kb.text(0.78, i + 0.5, f'C{octave}',
                   ha='center', va='center', fontsize=5.5,
                   color='#444', fontweight='bold', zorder=3)

ax_kb.set_xticks([])
ax_kb.set_yticks([])
for sp in ax_kb.spines.values():
    sp.set_visible(False)
ax_kb.patch.set_facecolor(PANEL)
ax_kb.axvline(1.0, color=SPINE, linewidth=1.2, zorder=4)

# ── Panel inferior: Piano-roll ────────────────────────────────────────────────
ax_roll.set_facecolor(PANEL)

for i in range(88):
    color = '#1c1c2e' if is_black(21 + i) else '#1e1e30'
    ax_roll.axhspan(i, i + 1, facecolor=color, alpha=1.0, zorder=0)

for midi_note in range(21, 109, 12):
    ax_roll.axhline(midi_note - 21, color=SPINE,
                    linewidth=0.45, linestyle='--', alpha=0.5, zorder=1)

cmap     = plt.get_cmap('YlOrRd')
norm_vel = Normalize(vmin=30, vmax=110)

for note in notes_in_seg:
    idx = note.pitch - 21
    if not (0 <= idx < 88):
        continue
    x0 = max(note.start, SEG_START)
    x1 = min(note.end,   SEG_END)
    if x1 - x0 <= 0:
        continue
    ax_roll.add_patch(mpatches.FancyBboxPatch(
        (x0, idx - 0.43), x1 - x0, 0.86,
        boxstyle='round,pad=0.04',
        facecolor=cmap(norm_vel(note.velocity)),
        edgecolor='white', linewidth=0.4, alpha=0.92, zorder=2))

ax_roll.set_xlim(SEG_START, SEG_END)
ax_roll.set_ylim(0, 88)

tick_t = np.arange(SEG_START, SEG_END + 0.5, 2)
ax_roll.set_xticks(tick_t)
ax_roll.set_xticklabels([f'{t:.0f} s' for t in tick_t],
                         color=TEXT, fontsize=8)
ax_roll.set_xlabel('Tiempo (s)', color=TEXT, fontsize=10)
ax_roll.set_yticks([])
ax_roll.tick_params(colors=TEXT)
for sp in ax_roll.spines.values():
    sp.set_edgecolor(SPINE)
ax_roll.spines['top'].set_edgecolor('#556688')
ax_roll.spines['top'].set_linewidth(1.2)

# ── Rejilla temporal compartida ───────────────────────────────────────────────
for t in tick_t:
    for ax in [ax_wave, ax_roll]:
        ax.axvline(t, color='#445566', linewidth=0.9,
                   linestyle='--', alpha=0.65, zorder=3)

# ── Línea de alineación destacada ─────────────────────────────────────────────
T_ALIGN = 1.82   # primer ataque visible
for ax in [ax_wave, ax_roll]:
    ax.axvline(T_ALIGN, color='#ffb74d', linewidth=1.6,
               linestyle='-', alpha=0.92, zorder=6)

ylo, yhi = ax_wave.get_ylim()
ax_wave.annotate(
    'Alineación temporal\n≤ 3 ms',
    xy=(T_ALIGN, ylo + (yhi - ylo) * 0.05),
    xytext=(T_ALIGN + 0.75, yhi * 0.60),
    color='#ffb74d', fontsize=7.5, fontstyle='italic',
    arrowprops=dict(arrowstyle='->', color='#ffb74d', lw=1.2),
    bbox=dict(facecolor=BG, edgecolor='#ffb74d',
              boxstyle='round,pad=0.25', linewidth=0.8)
)

# ── Etiquetas de paneles ──────────────────────────────────────────────────────
ax_wave.text(SEG_START + 0.15, yhi * 0.80,
             'Onda de audio', color='#90caf9',
             fontsize=8.5, fontweight='bold', va='top')
ax_roll.text(SEG_START + 0.15, 86,
             'Piano-roll MIDI', color='#ffcc80',
             fontsize=8.5, fontweight='bold', va='top', zorder=5)

# ── Barra de color ────────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([CBAR_LEFT, BOTTOM_ROLL, CBAR_WIDTH, HEIGHT_ROLL])
sm = ScalarMappable(cmap=cmap, norm=norm_vel)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('Velocidad MIDI', color=TEXT, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=7)

# ── Guardar ───────────────────────────────────────────────────────────────────
out = 'figures/audio_midi_alignment.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
print(f'Guardado: {out}')
plt.close()