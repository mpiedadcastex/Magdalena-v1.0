"""
Genera figures/stft_ejemplo.png  (forma de onda + espectrograma STFT)
        figures/logmel_ejemplo.png (espectrograma Log-Mel)
a partir de audio de piano sintetizado, sin necesitar un .wav externo.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("figures", exist_ok=True)

# ── Parámetros globales ──────────────────────────────────────────────────────
SR         = 16000
N_FFT      = 2048
HOP        = 320
N_MELS     = 229
FMIN       = 31
FMAX       = 8000

BG         = "#16213e"
PANEL      = "#1a1a2e"
TEXT       = "white"
CMAP_SPEC  = "magma"

# ── Síntesis de piano ────────────────────────────────────────────────────────
def midi_to_hz(midi_note: int) -> float:
    return 440.0 * 2 ** ((midi_note - 69) / 12)

def piano_note(freq: float, duration: float, velocity: float = 0.7) -> np.ndarray:
    """Síntesis aditiva con envolvente ADSR y caída por armónico."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Envolvente ADSR
    n_atk = max(1, int(0.006 * SR))
    n_dec = max(1, int(0.08  * SR))
    n_rel = max(1, int(min(0.25, duration * 0.25) * SR))
    n_sus = max(0, n - n_atk - n_dec - n_rel)

    env = np.concatenate([
        np.linspace(0.0,       1.0,       n_atk),
        np.linspace(1.0,       0.65,      n_dec),
        np.full(n_sus,         0.65),
        np.linspace(0.65,      0.0,       n_rel),
    ])[:n]

    # Suma de armónicos con caída temporal más rápida en los agudos
    signal = np.zeros(n)
    for k in range(1, 16):
        f_k = freq * k
        if f_k >= SR / 2:
            break
        amp   = velocity / (k ** 1.3)
        decay = np.exp(-t * (0.8 + k * 0.5))
        signal += amp * decay * np.sin(2 * np.pi * f_k * t)

    return signal * env


def build_audio() -> np.ndarray:
    """Construye ~11 s de piano con melodía + acordes."""
    total = int(SR * 11.5)
    buf   = np.zeros(total)

    def place(midi_note, start_s, dur_s, vel=0.7):
        freq  = midi_to_hz(midi_note)
        wave  = piano_note(freq, dur_s + 0.35, vel)   # +release extra
        i0    = int(start_s * SR)
        i1    = min(i0 + len(wave), total)
        buf[i0:i1] += wave[:i1 - i0]

    # Melodía mano derecha (Do mayor)
    melody = [
        (72, 0.0,  0.5, 0.70),
        (74, 0.5,  0.5, 0.65),
        (76, 1.0,  0.5, 0.72),
        (77, 1.5,  0.5, 0.68),
        (79, 2.0,  1.0, 0.75),
        (77, 3.0,  0.5, 0.62),
        (76, 3.5,  0.5, 0.60),
        (74, 4.0,  1.0, 0.65),
        (72, 5.0,  1.5, 0.70),
        (76, 6.5,  0.5, 0.58),
        (79, 7.0,  0.5, 0.68),
        (81, 7.5,  0.5, 0.72),
        (83, 8.0,  1.0, 0.75),
        (84, 9.0,  2.0, 0.80),
    ]

    # Acordes mano izquierda
    chords = [
        ([48, 52, 55], 0.0, 2.0, 0.50),   # Do mayor
        ([43, 47, 50], 2.0, 2.0, 0.48),   # Sol mayor
        ([45, 48, 52], 4.0, 2.0, 0.50),   # La menor
        ([43, 47, 50], 6.0, 2.0, 0.48),   # Sol mayor
        ([48, 52, 55], 8.0, 3.0, 0.52),   # Do mayor
    ]

    for midi_note, start, dur, vel in melody:
        place(midi_note, start, dur, vel)
    for pitches, start, dur, vel in chords:
        for p in pitches:
            place(p, start, dur, vel)

    # Normalizar
    peak = np.max(np.abs(buf))
    if peak > 0:
        buf /= peak
    return buf * 0.92


y = build_audio()

# ── Figura 1: forma de onda + espectrograma STFT ─────────────────────────────
S_mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
S_db  = librosa.amplitude_to_db(S_mag, ref=np.max)

fig = plt.figure(figsize=(12, 6), facecolor=BG)
gs  = gridspec.GridSpec(2, 1, hspace=0.45, top=0.93, bottom=0.09,
                        left=0.07, right=0.97)

# — Forma de onda —
ax0 = fig.add_subplot(gs[0])
ax0.set_facecolor(PANEL)
librosa.display.waveshow(y, sr=SR, ax=ax0, color="#4fc3f7", alpha=0.85)
ax0.set_xlabel("Tiempo (s)", color=TEXT, fontsize=10)
ax0.set_ylabel("Amplitud", color=TEXT, fontsize=10)
ax0.set_title("Forma de onda", color=TEXT, fontsize=11, pad=5)
ax0.tick_params(colors=TEXT, labelsize=8)
for sp in ax0.spines.values():
    sp.set_edgecolor("#444466")
ax0.yaxis.label.set_color(TEXT)
ax0.xaxis.label.set_color(TEXT)

# — Espectrograma STFT —
ax1 = fig.add_subplot(gs[1])
ax1.set_facecolor(PANEL)
img = librosa.display.specshow(
    S_db, sr=SR, hop_length=HOP,
    x_axis="time", y_axis="hz",
    fmin=FMIN, fmax=FMAX,
    ax=ax1, cmap=CMAP_SPEC
)
ax1.set_xlabel("Tiempo (s)", color=TEXT, fontsize=10)
ax1.set_ylabel("Frecuencia (Hz)", color=TEXT, fontsize=10)
ax1.set_title("Espectrograma de magnitudes (STFT)", color=TEXT, fontsize=11, pad=5)
ax1.tick_params(colors=TEXT, labelsize=8)
ax1.set_ylim(FMIN, FMAX)
for sp in ax1.spines.values():
    sp.set_edgecolor("#444466")

cbar1 = fig.colorbar(img, ax=ax1, pad=0.01, fraction=0.025)
cbar1.set_label("Energía (dB)", color=TEXT, fontsize=9)
cbar1.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar1.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)

plt.savefig("figures/stft_ejemplo.png", dpi=300, bbox_inches="tight", facecolor=BG)
print("Guardado: figures/stft_ejemplo.png")
plt.close()

# ── Figura 2: espectrograma Log-Mel ──────────────────────────────────────────
M = librosa.feature.melspectrogram(
    y=y, sr=SR,
    n_fft=N_FFT, hop_length=HOP,
    n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    power=2.0
)
M_db = librosa.power_to_db(M, ref=np.max)

fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)
ax.set_facecolor(PANEL)
fig.patch.set_facecolor(BG)

img2 = librosa.display.specshow(
    M_db, sr=SR, hop_length=HOP,
    x_axis="time", y_axis="mel",
    fmin=FMIN, fmax=FMAX,
    ax=ax, cmap=CMAP_SPEC
)
ax.set_xlabel("Tiempo (s)", color=TEXT, fontsize=10)
ax.set_ylabel("Frecuencia Mel (Hz)", color=TEXT, fontsize=10)
ax.set_title("Espectrograma Log-Mel  (229 filtros, 31 – 8 000 Hz)", color=TEXT,
             fontsize=11, pad=8)
ax.tick_params(colors=TEXT, labelsize=8)
for sp in ax.spines.values():
    sp.set_edgecolor("#444466")

cbar2 = fig.colorbar(img2, ax=ax, pad=0.01, fraction=0.025)
cbar2.set_label("Energía (dB)", color=TEXT, fontsize=9)
cbar2.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("figures/logmel_ejemplo.png", dpi=300, bbox_inches="tight", facecolor=BG)
print("Guardado: figures/logmel_ejemplo.png")
plt.close()
