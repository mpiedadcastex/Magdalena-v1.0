import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

# ── MIDI sintético ──────────────────────────────────────────────────────────
midi = pretty_midi.PrettyMIDI(initial_tempo=120)
piano = pretty_midi.Instrument(program=0)

def add_note(instr, pitch, start, end, velocity):
    instr.notes.append(pretty_midi.Note(
        velocity=velocity, pitch=pitch, start=start, end=end
    ))

melody = [
    (72, 0.0,  0.5, 90),
    (74, 0.5,  0.5, 85),
    (76, 1.0,  0.5, 92),
    (77, 1.5,  0.5, 88),
    (79, 2.0,  1.0, 95),
    (77, 3.0,  0.5, 82),
    (76, 3.5,  0.5, 80),
    (74, 4.0,  1.0, 85),
    (72, 5.0,  1.5, 90),
    (76, 6.5,  0.5, 78),
    (79, 7.0,  0.5, 88),
    (81, 7.5,  0.5, 92),
    (83, 8.0,  1.0, 95),
    (84, 9.0,  2.0, 100),
]

chords = [
    ([48, 52, 55], 0.0, 2.0, 65),
    ([43, 47, 50], 2.0, 2.0, 60),
    ([45, 48, 52], 4.0, 2.0, 62),
    ([43, 47, 50], 6.0, 2.0, 60),
    ([48, 52, 55], 8.0, 3.0, 65),
]

for pitch, start, dur, vel in melody:
    add_note(piano, pitch, start, start + dur, vel)
for pitches, start, dur, vel in chords:
    for p in pitches:
        add_note(piano, p, start, start + dur, vel)

midi.instruments.append(piano)

# ── Piano-roll (88 teclas: MIDI 21–108) ────────────────────────────────────
fs = 100
piano_roll = midi.get_piano_roll(fs=fs)[21:109, :]
total_time = midi.get_end_time()
T = piano_roll.shape[1]

# ── Figura ──────────────────────────────────────────────────────────────────
BG      = '#16213e'
PANEL   = '#1a1a2e'
WHITE_K = '#e8e8e8'
BLACK_K = '#111122'
GRID    = '#2a2a4a'
SPINE   = '#444466'

fig = plt.figure(figsize=(13, 5), facecolor=BG)
gs  = GridSpec(1, 2, width_ratios=[1, 13], wspace=0.0,
               left=0.01, right=0.93, top=0.90, bottom=0.12)
ax_kb   = fig.add_subplot(gs[0])
ax_roll = fig.add_subplot(gs[1])

# ── Teclado de piano ─────────────────────────────────────────────────────────
BLACK_SEMITONES = {1, 3, 6, 8, 10}   # C#, D#, F#, G#, A#

def is_black(midi_note):
    return (midi_note % 12) in BLACK_SEMITONES

ax_kb.set_facecolor(PANEL)
ax_kb.set_xlim(0, 1)
ax_kb.set_ylim(0, 88)

# 1) Teclas blancas (fondo completo, borde fino)
for i in range(88):
    if not is_black(21 + i):
        ax_kb.add_patch(mpatches.Rectangle(
            (0, i), 1.0, 1.0,
            facecolor=WHITE_K, edgecolor='#888', linewidth=0.35, zorder=1
        ))

# 2) Teclas negras encima (60 % de ancho)
for i in range(88):
    if is_black(21 + i):
        ax_kb.add_patch(mpatches.Rectangle(
            (0, i), 0.50, 1.0,
            facecolor=BLACK_K, edgecolor='#000', linewidth=0.3, zorder=2
        ))

# 3) Etiquetas C en teclas blancas
for i in range(88):
    midi_note = 21 + i
    if midi_note % 12 == 0:
        octave = midi_note // 12 - 1
        ax_kb.text(0.75, i + 0.5, f'C{octave}',
                   ha='center', va='center', fontsize=6, color='#444',
                   fontweight='bold', zorder=3)

ax_kb.set_xticks([])
ax_kb.set_yticks([])
for sp in ax_kb.spines.values():
    sp.set_visible(False)
# Línea separadora derecha
ax_kb.axvline(1.0, color=SPINE, linewidth=1.2, zorder=4)

# ── Piano-roll ───────────────────────────────────────────────────────────────
ax_roll.set_facecolor(PANEL)

# Rayas horizontales: negro en teclas negras, ligeramente más claro en blancas
for i in range(88):
    color = '#222235' if is_black(21 + i) else '#1e1e30'
    ax_roll.axhspan(i, i + 1, facecolor=color, alpha=1.0, zorder=0)

# Líneas guía por octava
for midi_note in range(21, 109, 12):
    ax_roll.axhline(midi_note - 21, color=SPINE, linewidth=0.5,
                    linestyle='--', alpha=0.5, zorder=1)

# Notas coloreadas por velocidad
cmap = plt.get_cmap('YlOrRd')
norm = Normalize(vmin=40, vmax=110)

for note in piano.notes:
    idx = note.pitch - 21
    if not (0 <= idx < 88):
        continue
    sf = int(note.start * fs)
    ef = int(note.end   * fs)
    w  = max(ef - sf, 1)
    ax_roll.add_patch(mpatches.FancyBboxPatch(
        (sf, idx - 0.44), w, 0.88,
        boxstyle='round,pad=0.05',
        facecolor=cmap(norm(note.velocity)),
        edgecolor='white', linewidth=0.5, alpha=0.93, zorder=2
    ))

ax_roll.set_xlim(0, T)
ax_roll.set_ylim(0, 88)

# Eje X en segundos
tick_t = np.arange(0, total_time + 1, 1)
ax_roll.set_xticks((tick_t * fs).astype(int))
ax_roll.set_xticklabels([f'{t:.0f}s' for t in tick_t],
                         color='white', fontsize=9)
ax_roll.set_xlabel('Tiempo (s)', color='white', fontsize=11)

# Eje Y sin etiquetas (el teclado ya las da)
ax_roll.set_yticks([])
ax_roll.tick_params(colors='white')
for sp in ax_roll.spines.values():
    sp.set_edgecolor(SPINE)

ax_roll.set_title(
    'Piano-roll — tiempo / semitono / dinámica',
    color='white', fontsize=13, pad=8
)

# ── Barra de color ───────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_roll, orientation='vertical',
                    pad=0.015, fraction=0.025)
cbar.set_label('Velocidad MIDI\n(dinámica)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

# ── Guardar ──────────────────────────────────────────────────────────────────
import os; os.makedirs('figures', exist_ok=True)
plt.savefig('figures/piano_roll_ejemplo.png', dpi=300,
            bbox_inches='tight', facecolor=BG)
print('Guardado en figures/piano_roll_ejemplo.png')
