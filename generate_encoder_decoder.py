"""
Genera figures/encoder_decoder.png (300 dpi)
Diagrama Encoder-Decoder con matplotlib puro.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
os.makedirs('figures', exist_ok=True)

# ── Paleta ───────────────────────────────────────────────────────────────────
EF, EE, ET = '#dbeafe', '#3b82f6', '#1e3a8a'      # encoder
DF, DE, DT = '#ffedd5', '#f97316', '#7c2d12'      # decoder
MF, ME     = '#bfdbfe', '#2563eb'                 # memoria
IOF, IOE, IOT = '#f3f4f6', '#9ca3af', '#374151'   # IO nodes
GEF, GEE   = '#eff6ff', '#3b82f6'                 # encoder group
GDF, GDE   = '#fff7ed', '#f97316'                 # decoder group
CA_C       = '#0d9488'                            # cross-attention
DASH_C     = '#9ca3af'
ARR_C      = '#1f2937'

# ── Geometría ────────────────────────────────────────────────────────────────
ENC_CX, DEC_CX = 3.2, 10.8
BW   = 3.8    # ancho de caja
BH   = 0.72   # alto caja regular
BH2  = 0.96   # alto caja dos líneas
BHIO = 0.56   # alto nodo IO
RP   = 0.10   # radio de esquinas (boxstyle pad)

# Posiciones Y encoder (de abajo a arriba)
Y_EI = 1.8    # input IO
Y_EP = 3.55   # Proyección (two-line)
Y_ES = 5.15   # Autoatención
Y_EF = 6.35   # Feed-Forward
Y_EM = 7.80   # Memoria

# Posiciones Y decoder (de abajo a arriba)
Y_DI = 1.8    # input IO
Y_DE = 3.55   # Embedding (two-line)
Y_DS = 5.15   # Autoatención causal
Y_DC = 6.35   # Atención cruzada
Y_DF = 7.55   # Feed-Forward
Y_DP = 9.10   # Proyección lineal (two-line)
Y_DO = 11.2   # output IO

# ── Figura ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14 / 2.54, 18 / 2.54))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 12.5)
ax.axis('off')

# ── Helpers ───────────────────────────────────────────────────────────────────
def rounded_box(cx, cy, w, h, fill, edge, text, tcolor,
                fs=9.5, bold=False, lw=1.5, z=3):
    """Caja redondeada con texto centrado."""
    p = RP
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2 + p, cy - h/2 + p), w - 2*p, h - 2*p,
        boxstyle=f'round,pad={p}',
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z
    ))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tcolor, fontweight='bold' if bold else 'normal',
            multialignment='center', linespacing=1.3, zorder=z + 1)

def io_node(cx, cy, text):
    """Nodo IO con borde gris."""
    p = 0.07
    w, h = BW + 0.4, BHIO
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2 + p, cy - h/2 + p), w - 2*p, h - 2*p,
        boxstyle=f'round,pad={p}',
        facecolor=IOF, edgecolor=IOE, linewidth=1.3, zorder=3
    ))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=8.5,
            color=IOT, multialignment='center', linespacing=1.2, zorder=4)

def arrow(x0, y0, x1, y1, c=ARR_C, lw=1.3, ls='solid', ms=13, z=6):
    """Flecha sencilla con punta rellena."""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=c, lw=lw, linestyle=ls,
                    mutation_scale=ms,
                ), zorder=z)

def group_box(x0, y0, x1, y1, fill, edge, lw=1.8, z=1):
    """Rectángulo de grupo redondeado."""
    p = 0.18
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0 + p, y0 + p), (x1 - x0) - 2*p, (y1 - y0) - 2*p,
        boxstyle=f'round,pad={p}',
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z, alpha=0.92
    ))

def dashed_block(x0, y0, x1, y1, z=2):
    """Rectángulo punteado para bloque repetido."""
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        facecolor='none', edgecolor=DASH_C,
        linewidth=1.1, linestyle='--', zorder=z
    ))

# ── Cajas de grupo (se dibujan primero, quedan al fondo) ──────────────────────
PAD_G = 0.42   # padding alrededor del contenido dentro del grupo

enc_gx0 = ENC_CX - BW/2 - PAD_G
enc_gx1 = ENC_CX + BW/2 + PAD_G
enc_gy0 = Y_EP - BH2/2 - PAD_G
enc_gy1 = Y_EM + BH/2  + PAD_G
group_box(enc_gx0, enc_gy0, enc_gx1, enc_gy1, GEF, GEE)

dec_gx0 = DEC_CX - BW/2 - PAD_G
dec_gx1 = DEC_CX + BW/2 + PAD_G
dec_gy0 = Y_DE - BH2/2 - PAD_G
dec_gy1 = Y_DP + BH2/2 + PAD_G
group_box(dec_gx0, dec_gy0, dec_gx1, dec_gy1, GDF, GDE)

# ── Bloques punteados de repetición ──────────────────────────────────────────
PAD_D = 0.26

# Encoder: Autoatención + Feed-Forward
dashed_block(
    ENC_CX - BW/2 - PAD_D,
    Y_ES - BH/2 - PAD_D,
    ENC_CX + BW/2 + PAD_D,
    Y_EF + BH/2 + PAD_D,
)
ax.text(ENC_CX + BW/2 + PAD_D + 0.12,
        (Y_ES + Y_EF) / 2,
        r'$\times\, L_{enc}$',
        ha='left', va='center', fontsize=8.5, color=DASH_C, zorder=6)

# Decoder: Autoatención causal + Atención cruzada + Feed-Forward
dashed_block(
    DEC_CX - BW/2 - PAD_D,
    Y_DS - BH/2 - PAD_D,
    DEC_CX + BW/2 + PAD_D,
    Y_DF + BH/2 + PAD_D,
)
ax.text(DEC_CX + BW/2 + PAD_D + 0.12,
        (Y_DS + Y_DF) / 2,
        r'$\times\, L_{dec}$',
        ha='left', va='center', fontsize=8.5, color=DASH_C, zorder=6)

# ── Nodos del codificador ─────────────────────────────────────────────────────
io_node(ENC_CX, Y_EI, 'Espectrograma Log-Mel')
rounded_box(ENC_CX, Y_EP, BW, BH2, EF, EE,
            'Proyección lineal\n+ Pos. Enc. aprendido', ET, fs=9)
rounded_box(ENC_CX, Y_ES, BW, BH,  EF, EE, 'Autoatención', ET)
rounded_box(ENC_CX, Y_EF, BW, BH,  EF, EE, 'Feed-Forward', ET)
rounded_box(ENC_CX, Y_EM, BW, BH,  MF, ME, 'Memoria', ET, bold=True, lw=2.0)

# ── Flechas internas del codificador ─────────────────────────────────────────
arrow(ENC_CX, Y_EI + BHIO/2,  ENC_CX, Y_EP - BH2/2)
arrow(ENC_CX, Y_EP + BH2/2,   ENC_CX, Y_ES - BH/2)
arrow(ENC_CX, Y_ES + BH/2,    ENC_CX, Y_EF - BH/2)
arrow(ENC_CX, Y_EF + BH/2,    ENC_CX, Y_EM - BH/2)

# ── Nodos del decodificador ───────────────────────────────────────────────────
io_node(DEC_CX, Y_DI, 'Tokens MIDI previos')
rounded_box(DEC_CX, Y_DE, BW, BH2, DF, DE,
            'Embedding\n+ Pos. Enc. sinusoidal', DT, fs=9)
rounded_box(DEC_CX, Y_DS, BW, BH,  DF, DE, 'Autoatención causal', DT)
rounded_box(DEC_CX, Y_DC, BW, BH,  DF, DE, 'Atención cruzada', DT)
rounded_box(DEC_CX, Y_DF, BW, BH,  DF, DE, 'Feed-Forward', DT)
rounded_box(DEC_CX, Y_DP, BW, BH2, DF, DE,
            'Proyección lineal\n(311 clases)', DT, fs=9)
io_node(DEC_CX, Y_DO, 'Token MIDI siguiente')

# ── Flechas internas del decodificador ───────────────────────────────────────
arrow(DEC_CX, Y_DI + BHIO/2,  DEC_CX, Y_DE - BH2/2)
arrow(DEC_CX, Y_DE + BH2/2,   DEC_CX, Y_DS - BH/2)
arrow(DEC_CX, Y_DS + BH/2,    DEC_CX, Y_DC - BH/2)
arrow(DEC_CX, Y_DC + BH/2,    DEC_CX, Y_DF - BH/2)
arrow(DEC_CX, Y_DF + BH/2,    DEC_CX, Y_DP - BH2/2)
arrow(DEC_CX, Y_DP + BH2/2,   DEC_CX, Y_DO - BHIO/2)

# ── Flecha de atención cruzada (codo, teal discontinuo) ───────────────────────
x_mem_r = ENC_CX + BW/2   # borde derecho de Memoria
x_ca_l  = DEC_CX - BW/2   # borde izquierdo de Atención cruzada
MID_X   = (x_mem_r + x_ca_l) / 2   # punto de codo horizontal
lw_ca   = 1.7

# Segmento horizontal desde Memoria
ax.plot([x_mem_r, MID_X], [Y_EM, Y_EM],
        color=CA_C, lw=lw_ca, ls='--', zorder=5,
        solid_capstyle='round')
# Segmento vertical bajando
ax.plot([MID_X, MID_X], [Y_EM, Y_DC],
        color=CA_C, lw=lw_ca, ls='--', zorder=5,
        solid_capstyle='round')
# Segmento horizontal con flecha hacia Atención cruzada
ax.annotate('', xy=(x_ca_l, Y_DC), xytext=(MID_X, Y_DC),
            arrowprops=dict(
                arrowstyle='-|>',
                color=CA_C, lw=lw_ca, linestyle='--',
                mutation_scale=13,
            ), zorder=6)

# Etiquetas de la flecha cruzada
ax.text(MID_X, Y_EM + 0.20,
        'K, V',
        ha='center', va='bottom', fontsize=8.5, color=CA_C,
        fontweight='bold', zorder=7)
ax.text(MID_X + 0.18, (Y_EM + Y_DC) / 2,
        'atención\ncruzada',
        ha='left', va='center', fontsize=8, color=CA_C,
        fontstyle='italic', linespacing=1.25, zorder=7)

# ── Etiquetas de grupo ────────────────────────────────────────────────────────
ax.text(ENC_CX, enc_gy1 + 0.18,
        'CODIFICADOR',
        ha='center', va='bottom', fontsize=10.5,
        color=GEE, fontweight='bold', zorder=7)

ax.text(DEC_CX, dec_gy1 + 0.18,
        'DECODIFICADOR',
        ha='center', va='bottom', fontsize=10.5,
        color=GDE, fontweight='bold', zorder=8,
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.18'))

# ── Guardar ───────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.2)
plt.savefig('figures/encoder_decoder.png', dpi=300, bbox_inches='tight')
print('Guardado: figures/encoder_decoder.png')
plt.close()
