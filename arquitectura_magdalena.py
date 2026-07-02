"""
arquitectura_magdalena.py — Diagrama de arquitectura Magdalena v1.0
Salida: figs/arquitectura_global.pdf  (18 cm × 22 cm, 300 dpi)
Layout: tres columnas, flujo de abajo hacia arriba.
  Izquierda  → Codificador
  Centro     → Bloque Sparsifiner detallado (×6)
  Derecha    → Decodificador
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

os.makedirs("figs", exist_ok=True)

# ── Paleta ──────────────────────────────────────────────────────────────────
C_ENC_F,  C_ENC_E  = "#dbeafe", "#3b82f6"   # encoder  – azul claro
C_MASK_F, C_MASK_E = "#fef3c7", "#d97706"   # mask     – ámbar
C_DEC_F,  C_DEC_E  = "#ffedd5", "#f97316"   # decoder  – naranja
C_MEM_F,  C_MEM_E  = "#bfdbfe", "#1d4ed8"   # memoria  – azul oscuro
C_IO_F,   C_IO_E   = "#f3f4f6", "#9ca3af"   # I/O      – gris
C_ARR = "#333333"      # negro ~70 %
C_KV  = "#0d9488"      # teal para K, V
C_RES = "#adb5bd"      # gris claro residual

FS = 9   # fuente base (pt)

# ── Figura 18 × 22 cm ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18 / 2.54, 22 / 2.54))
ax.set_xlim(0, 18)
ax.set_ylim(0, 22)
ax.axis("off")
plt.rcParams.update({"font.family": "sans-serif", "font.size": FS})

# ── Primitivas ───────────────────────────────────────────────────────────────

def blk(cx, cy, w, h, text, fc, ec,
        lw=1.0, ls="-", fs=FS, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.07",
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            multialignment="center", color=tc, zorder=4)


def arr(x1, y1, x2, y2, c=C_ARR, lw=1.2, dashed=False, ms=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>", color=c, lw=lw,
                    linestyle=(0, (4, 3)) if dashed else "solid",
                    mutation_scale=ms),
                zorder=5)


def seg(xs, ys, c=C_ARR, lw=1.2, dashed=False):
    ax.plot(xs, ys, c=c, lw=lw,
            ls=(0, (4, 3)) if dashed else "solid", zorder=4)


def plus_node(cx, cy, r=0.23):
    ax.add_patch(plt.Circle(
        (cx, cy), r, fc="white", ec=C_RES, lw=0.9, zorder=4))
    ax.text(cx, cy, "+", ha="center", va="center",
            fontsize=FS - 1, color=C_RES, zorder=5)


def dashed_container(x, y, w, h, label, ec):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        fc="none", ec=ec, lw=0.9, ls=(0, (5, 3)), zorder=2))
    ax.text(x + w / 2, y + h + 0.1, label,
            ha="center", va="bottom",
            fontsize=FS - 0.5, style="italic", color=ec, zorder=4)


# ── Coordenadas de columna ───────────────────────────────────────────────────
XE, XS, XD = 3.0, 9.0, 15.0        # centros
WE, WS, WD, WIO = 4.8, 5.4, 4.8, 4.3   # anchos
HS   = 1.05    # alto bloque estándar
HIO  = 0.85    # alto I/O
HLN  = 0.65    # alto LayerNorm
HMP  = 2.55    # alto MaskPredictor


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMNA IZQUIERDA — CODIFICADOR
# ═══════════════════════════════════════════════════════════════════════════════

ax.text(XE, 21.2, "CODIFICADOR", ha="center", va="center",
        fontsize=FS + 1.5, fontweight="bold", color=C_ENC_E)

# Entrada (I/O, fuera de cajas coloreadas)
YEI = 1.15
blk(XE, YEI, WIO, HIO,
    "Espectrograma Log-Mel\n(B × 229 × T)", C_IO_F, C_IO_E, fs=FS - 0.5)

# Proyección lineal + PosEnc aprendido
YEP = 3.1
arr(XE, YEI + HIO / 2, XE, YEP - HS / 2)
blk(XE, YEP, WE, HS,
    "Proyección lineal  +  Pos. Enc. aprendido", C_ENC_F, C_ENC_E)

# Conexión Proyección → base del container Sparsifiner con anotación ×6
# Entrada por la base del container refleja el flujo vertical (abajo→arriba)
YCB_ENTRY = 4.2                              # nivel de entrada, justo bajo YCB=4.35
arr(XE, YEP + HS / 2, XE, YCB_ENTRY)        # tramo vertical desde Proyección
seg([XE, XS], [YCB_ENTRY, YCB_ENTRY])        # tramo horizontal hasta la base del container
ax.text((XE + XS) / 2, YCB_ENTRY - 0.15, "×6",
        ha="center", va="top", fontsize=FS, color=C_ARR)

# Memoria M — salida del encoder (esquina superior izquierda)
YEM = 17.8
blk(XE, YEM, WE, HS + 0.2,
    "Memoria  M\n(B × T × d)",
    C_MEM_F, C_MEM_E, lw=1.8, bold=True)


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMNA CENTRAL — BLOQUE SPARSIFINER
# ═══════════════════════════════════════════════════════════════════════════════

YCB, YCT = 4.35, 19.5
dashed_container(XS - WS / 2 - 0.48, YCB, WS + 0.96, YCT - YCB,
                 "Bloque Sparsifiner  (×6)", ec=C_ENC_E)

# ── LN 1 ──────────────────────────────────────────────────────────────────────
# Entrada al container por la base: la flecha sube verticalmente desde YCB_ENTRY
YLN1 = 5.5
arr(XS, YCB_ENTRY, XS, YLN1 - HLN / 2, lw=1.1)
blk(XS, YLN1, 2.3, HLN, "LN", C_IO_F, C_IO_E, fs=FS - 1)

# ── MaskPredictor ─────────────────────────────────────────────────────────────
YMP = 8.15
arr(XS, YLN1 + HLN / 2, XS, YMP - HMP / 2, lw=1.0)
blk(XS, YMP, WS - 0.12, HMP,
    "MaskPredictor\n" + "─" * 30 + "\n"
    "Compresión canal  (r_c = 2)\n"
    "Compresión temporal  (r_N = 64)\n"
    "Cheap Attention  →  Top-k",
    C_MASK_F, C_MASK_E, lw=1.3, fs=FS - 0.5)

# ── SparseAttn ────────────────────────────────────────────────────────────────
YSA = 10.85
arr(XS, YMP + HMP / 2, XS, YSA - HS / 2, lw=1.0)
blk(XS, YSA, WS - 0.12, HS,
    "SparseAttn — atención dispersa\nO(r_attn · T²)", C_ENC_F, C_ENC_E, fs=FS - 0.5)

# ── DropPath 1 (entre SparseAttn y + residual 1) ─────────────────────────────
HDP2 = 0.55
YDP2 = YSA + HS / 2 + 0.38        # justo sobre SparseAttn
arr(XS, YSA + HS / 2, XS, YDP2 - HDP2 / 2, lw=1.0)
blk(XS, YDP2, 2.8, HDP2, "DropPath", C_IO_F, C_IO_E, fs=FS - 1.5)

# ── + residual 1  (a la derecha, gris) ───────────────────────────────────────
YP1 = YDP2 + HDP2 / 2 + 0.5
XRR = XS + WS / 2 + 0.52          # x de la línea de skip derecha
arr(XS, YDP2 + HDP2 / 2, XS, YP1 - 0.23, lw=1.0)
plus_node(XS, YP1)
# Skip: nace en el flujo central justo antes de LN₁ (nodo de bifurcación) → borde derecho → hasta +₁
Y_SK1 = YLN1 - HLN / 2 - 0.15    # punto de bifurcación sobre la línea de flujo, antes de LN₁
ax.add_patch(plt.Circle((XS, Y_SK1), 0.07, fc=C_RES, ec=C_RES, zorder=5))
seg([XS, XRR], [Y_SK1, Y_SK1], c=C_RES, lw=0.9)
seg([XRR, XRR], [Y_SK1, YP1], c=C_RES, lw=0.9)
arr(XRR, YP1, XS + 0.23, YP1, c=C_RES, lw=0.9, ms=6)

# ── LN 2 ──────────────────────────────────────────────────────────────────────
YLN2 = YP1 + 0.23 + 0.58
arr(XS, YP1 + 0.23, XS, YLN2 - HLN / 2, lw=1.0)
blk(XS, YLN2, 2.3, HLN, "LN", C_IO_F, C_IO_E, fs=FS - 1)

# ── FFN ───────────────────────────────────────────────────────────────────────
YFFN = YLN2 + HLN / 2 + 0.58 + HS / 2
arr(XS, YLN2 + HLN / 2, XS, YFFN - HS / 2, lw=1.0)
blk(XS, YFFN, WS - 0.12, HS,
    "FFN  (GELU,  d_ff = 2048)", C_ENC_F, C_ENC_E, fs=FS - 0.5)

# ── DropPath 2 (entre FFN y + residual 2 — orden correcto: FFN → DropPath₂ → +) ─
YDP = YFFN + HS / 2 + 0.38
arr(XS, YFFN + HS / 2, XS, YDP - HLN / 2, lw=1.0)
blk(XS, YDP, 3.2, HLN, "DropPath", C_IO_F, C_IO_E, fs=FS - 1)

# ── + residual 2  (a la derecha, gris) ───────────────────────────────────────
YP2 = YDP + HLN / 2 + 0.5
arr(XS, YDP + HLN / 2, XS, YP2 - 0.23, lw=1.0)
plus_node(XS, YP2)
# Skip: nace en el flujo central justo antes de LN₂ (nodo de bifurcación) → borde derecho → hasta +₂
Y_SK2 = YLN2 - HLN / 2 - 0.15    # punto de bifurcación sobre la línea de flujo, antes de LN₂
ax.add_patch(plt.Circle((XS, Y_SK2), 0.07, fc=C_RES, ec=C_RES, zorder=5))
seg([XS, XRR], [Y_SK2, Y_SK2], c=C_RES, lw=0.9)
seg([XRR, XRR], [Y_SK2, YP2], c=C_RES, lw=0.9)
arr(XRR, YP2, XS + 0.23, YP2, c=C_RES, lw=0.9, ms=6)

# Salida del Sparsifiner → Memoria M (arriba-izquierda)
YSO = YP2 + 0.23 + 0.35
arr(XS, YP2 + 0.23, XS, YSO, lw=1.1)
# Ruta: sube hasta la altura de Memoria, luego izquierda
seg([XS, XS], [YSO, YEM], lw=1.1)
arr(XS, YEM, XE + WE / 2, YEM, lw=1.1)


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMNA DERECHA — DECODIFICADOR
# ═══════════════════════════════════════════════════════════════════════════════

ax.text(XD, 21.2, "DECODIFICADOR", ha="center", va="center",
        fontsize=FS + 1.5, fontweight="bold", color=C_DEC_E)

# Entrada (I/O, fuera de cajas coloreadas)
YDI = 1.15
blk(XD, YDI, WIO, HIO,
    "Tokens MIDI previos", C_IO_F, C_IO_E, fs=FS - 0.5)

# Embedding + PosEnc sinusoidal
YDE = 3.1
arr(XD, YDI + HIO / 2, XD, YDE - HS / 2)
blk(XD, YDE, WD, HS,
    "Embedding  +  Pos. Enc. sinusoidal", C_DEC_F, C_DEC_E)

# Container capas decoder ×6
YDCB, YDCT = 4.35, 16.5
dashed_container(XD - WD / 2 - 0.45, YDCB, WD + 0.9, YDCT - YDCB,
                 "Capa Transformer  (×6)", ec=C_DEC_E)
arr(XD, YDE + HS / 2, XD, YDCB + 0.4, lw=1.0)

# ── Autoatención causal ───────────────────────────────────────────────────────
YDSA = 6.3
arr(XD, YDCB + 0.4, XD, YDSA - HS / 2, lw=1.0)
blk(XD, YDSA, WD - 0.2, HS,
    "Autoatención causal\n(máscara triangular)", C_DEC_F, C_DEC_E, fs=FS - 0.5)

# ── Atención cruzada ──────────────────────────────────────────────────────────
YDCA = 10.0
arr(XD, YDSA + HS / 2, XD, YDCA - HS / 2, lw=1.0)
blk(XD, YDCA, WD - 0.2, HS,
    "Atención cruzada", C_DEC_F, C_DEC_E, fs=FS - 0.5)

# ── FFN decoder ───────────────────────────────────────────────────────────────
YDFF = 13.7
arr(XD, YDCA + HS / 2, XD, YDFF - HS / 2, lw=1.0)
blk(XD, YDFF, WD - 0.2, HS,
    "FFN  (ReLU,  d_ff = 2048)", C_DEC_F, C_DEC_E, fs=FS - 0.5)

# Salida container
arr(XD, YDFF + HS / 2, XD, YDCT - 0.2, lw=1.0)
arr(XD, YDCT - 0.1, XD, YDCT + 0.35, lw=1.0)

# Proyección lineal (311 clases)
YDPJ = 17.8
arr(XD, YDCT + 0.35, XD, YDPJ - HS / 2, lw=1.0)
blk(XD, YDPJ, WD, HS,
    "Proyección lineal\n(311 clases)", C_DEC_F, C_DEC_E)

# Salida (I/O, fuera de cajas coloreadas)
YDIO = 19.65
arr(XD, YDPJ + HS / 2, XD, YDIO - HIO / 2, lw=1.0)
blk(XD, YDIO, WIO, HIO,
    "Token MIDI siguiente", C_IO_F, C_IO_E, fs=FS - 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# FLECHAS K, V  (Memoria → Atención cruzada, teal discontinua)
# Ruta: sube desde Memoria → arco por encima del container → baja hasta Atención cruzada
# ═══════════════════════════════════════════════════════════════════════════════
Y_KV_ARCH = 20.35    # altura sobre el container Sparsifiner (YCT=19.5)

# Tramo 1: de Memoria M arriba hasta Y_KV_ARCH
seg([XE, XE], [YEM + (HS + 0.2) / 2, Y_KV_ARCH], c=C_KV, lw=1.3, dashed=True)
# Tramo 2: horizontal sobre el container hasta el centro del decoder
seg([XE, XD], [Y_KV_ARCH, Y_KV_ARCH], c=C_KV, lw=1.3, dashed=True)
# Tramo 3: bajada con flecha hasta Atención cruzada
arr(XD, Y_KV_ARCH, XD, YDCA + HS / 2, c=C_KV, lw=1.3, dashed=True, ms=8)

# Etiqueta K, V al lado del tramo vertical derecho
ax.text(XD + 0.25, (Y_KV_ARCH + YDCA) / 2,
        "K, V", ha="left", va="center",
        fontsize=FS - 0.5, color=C_KV, style="italic")


# ── Guardar ──────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.1)
fig.savefig("figs/arquitectura_global.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✓  figs/arquitectura_global.pdf guardado")
