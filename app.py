import io
import os
import re
import sys
import tempfile

import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

import streamlit as st

# Añadir la raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.audio_proc import AudioProcessor
from src.data.midi_proc import MidiProcessor
from src.models.transformer import PianoTranscriptionModel
from utils import get_model_config

# ── Constantes ────────────────────────────────────────────────────────────────
BG       = "#16213e"
PANEL    = "#1a1a2e"
ACCENT   = "#4fc3f7"
SPINE    = "#444466"
WHITE_K  = "#e8e8e8"
BLACK_K  = "#111122"
GRID     = "#2a2a4a"

MAX_AUDIO_FRAMES = 4096   # ~82 s a 50 fps
SAMPLE_RATE      = 16000
HOP_LENGTH       = 320    # 50 fps
FPS              = SAMPLE_RATE // HOP_LENGTH  # 50

BLACK_SEMITONES = {1, 3, 6, 8, 10}

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Magdalena v1.0",
    page_icon="🎹",
    layout="centered",
)

# ── Cabecera ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        background-color:{BG};
        border-left: 5px solid {ACCENT};
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    ">
        <h1 style="color:{ACCENT}; margin:0; font-size:1.9rem;">
            🎹&nbsp; Magdalena v1.0 — Transcripción automática de piano
        </h1>
        <p style="color:#ffffff; margin:0.4rem 0 0 0; font-size:1rem;">
            Sube un archivo WAV y obtén la transcripción MIDI
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_black(midi_note: int) -> bool:
    return (midi_note % 12) in BLACK_SEMITONES


def _find_latest_checkpoint() -> tuple[str | None, int]:
    """Devuelve (ruta_checkpoint, epoch) o (None, -1) si no hay ninguno."""
    if not os.path.isdir("checkpoints"):
        return None, -1

    best_epoch = -1
    best_path  = None

    for root, _dirs, files in os.walk("checkpoints"):
        for fname in files:
            m = re.search(r"model_epoch_(\d+)\.pth$", fname)
            if m:
                epoch = int(m.group(1))
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_path  = os.path.join(root, fname)

    return best_path, best_epoch


def _mpl_dark_axes(ax, fig):
    """Aplica el estilo oscuro del proyecto a un par (fig, ax)."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)


def _plot_waveform(audio: np.ndarray, sr: int) -> plt.Figure:
    duration = len(audio) / sr
    times    = np.linspace(0, duration, len(audio))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    _mpl_dark_axes(ax, fig)
    ax.plot(times, audio, color=ACCENT, linewidth=0.4, alpha=0.85)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_title("Forma de onda")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_spectrogram(mel: np.ndarray, duration: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2.5))
    _mpl_dark_axes(ax, fig)
    im = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, duration, 0, mel.shape[0]],
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalizado", color="white", fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="white")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Bandas Mel")
    ax.set_title("Espectrograma Log-Mel")
    fig.tight_layout()
    return fig


def _plot_piano_roll(notes: list, total_time: float) -> plt.Figure:
    """Piano roll idéntico al de generate_piano_roll_2.py."""
    fs   = 100   # resolución interna: 100 frames/s para los rectángulos
    cmap = plt.get_cmap("YlOrRd")
    norm = Normalize(vmin=40, vmax=110)

    T = int(total_time * fs) + 1

    fig = plt.figure(figsize=(12, 4.5), facecolor=BG)
    gs  = GridSpec(1, 2, width_ratios=[1, 13], wspace=0.0,
                   left=0.01, right=0.93, top=0.88, bottom=0.14)
    ax_kb   = fig.add_subplot(gs[0])
    ax_roll = fig.add_subplot(gs[1])

    # — Teclado —
    ax_kb.set_facecolor(PANEL)
    ax_kb.set_xlim(0, 1)
    ax_kb.set_ylim(0, 88)

    for i in range(88):
        if not _is_black(21 + i):
            ax_kb.add_patch(mpatches.Rectangle(
                (0, i), 1.0, 1.0,
                facecolor=WHITE_K, edgecolor="#888", linewidth=0.3, zorder=1,
            ))
    for i in range(88):
        if _is_black(21 + i):
            ax_kb.add_patch(mpatches.Rectangle(
                (0, i), 0.50, 1.0,
                facecolor=BLACK_K, edgecolor="#000", linewidth=0.3, zorder=2,
            ))
    for i in range(88):
        midi_note = 21 + i
        if midi_note % 12 == 0:
            octave = midi_note // 12 - 1
            ax_kb.text(0.75, i + 0.5, f"C{octave}",
                       ha="center", va="center", fontsize=5.5,
                       color="#444", fontweight="bold", zorder=3)

    ax_kb.set_xticks([])
    ax_kb.set_yticks([])
    for sp in ax_kb.spines.values():
        sp.set_visible(False)
    ax_kb.axvline(1.0, color=SPINE, linewidth=1.2, zorder=4)

    # — Roll —
    ax_roll.set_facecolor(PANEL)

    for i in range(88):
        color = "#222235" if _is_black(21 + i) else "#1e1e30"
        ax_roll.axhspan(i, i + 1, facecolor=color, alpha=1.0, zorder=0)

    for midi_note in range(21, 109, 12):
        ax_roll.axhline(midi_note - 21, color=SPINE, linewidth=0.5,
                        linestyle="--", alpha=0.5, zorder=1)

    for note in notes:
        idx = note.pitch - 21
        if not (0 <= idx < 88):
            continue
        sf = int(note.start * fs)
        ef = int(note.end   * fs)
        w  = max(ef - sf, 1)
        ax_roll.add_patch(mpatches.FancyBboxPatch(
            (sf, idx - 0.44), w, 0.88,
            boxstyle="round,pad=0.05",
            facecolor=cmap(norm(note.velocity)),
            edgecolor="white", linewidth=0.4, alpha=0.93, zorder=2,
        ))

    ax_roll.set_xlim(0, T)
    ax_roll.set_ylim(0, 88)

    tick_t = np.arange(0, total_time + 1, max(1, int(total_time / 10)))
    ax_roll.set_xticks((tick_t * fs).astype(int))
    ax_roll.set_xticklabels([f"{t:.0f}s" for t in tick_t], color="white", fontsize=8)
    ax_roll.set_xlabel("Tiempo (s)", color="white", fontsize=10)
    ax_roll.set_yticks([])
    ax_roll.tick_params(colors="white")
    for sp in ax_roll.spines.values():
        sp.set_edgecolor(SPINE)
    ax_roll.set_title("Piano-roll  ·  tiempo / semitono / dinámica",
                      color="white", fontsize=12, pad=6)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_roll, orientation="vertical",
                        pad=0.015, fraction=0.025)
    cbar.set_label("Velocidad MIDI", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    return fig


def _predict_sampling(
    model: PianoTranscriptionModel,
    audio_tensor: torch.Tensor,
    mp: MidiProcessor,
    device: torch.device,
    temperature: float = 0.8,
    max_len: int = 1500,
    progress_bar=None,
) -> list[int]:
    """Temperature sampling sin dependencia de tqdm."""
    model.eval()
    sos = mp.token_sos
    eos = mp.token_eos

    generated = torch.tensor([[sos]], dtype=torch.long, device=device)
    tokens    = []

    with torch.no_grad():
        for step in range(max_len):
            logits     = model(audio_tensor, generated, tgt_padding_mask=None)
            last_logits = logits[:, -1, :] / temperature
            probs       = torch.softmax(last_logits, dim=-1)
            next_token  = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            if token_id == eos:
                break

            tokens.append(token_id)
            generated = torch.cat([generated, next_token], dim=1)

            if progress_bar is not None:
                progress_bar.progress(min((step + 1) / max_len, 1.0))

            if step % 100 == 0:
                torch.cuda.empty_cache()

    return tokens


# ── Carga del modelo (una sola vez) ──────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    """Devuelve (model | None, midi_processor, epoch_cargado)."""
    mp  = MidiProcessor()
    ckpt_path, epoch = _find_latest_checkpoint()

    if ckpt_path is None:
        return None, mp, -1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = get_model_config()

    model = PianoTranscriptionModel(
        midi_processor=mp,
        encoder_cfg=cfg,
        embed_dim=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        nhead=4,
    ).to(device)

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, mp, epoch
    except Exception as exc:
        st.error(f"Error al cargar el checkpoint `{ckpt_path}`: {exc}")
        return None, mp, -1


model, midi_processor, loaded_epoch = load_model()

# ── Estado del checkpoint ─────────────────────────────────────────────────────
if model is None:
    st.warning(
        "⚠️ No se encontró ningún checkpoint entrenado. "
        "Ejecuta primero `train.py`"
    )
else:
    st.success(f"Modelo listo — epoch {loaded_epoch}")

st.divider()

# ── Panel de carga ────────────────────────────────────────────────────────────
st.subheader("Cargar audio")
uploaded = st.file_uploader("Selecciona un archivo WAV", type=["wav"])

if uploaded is not None:
    # Guardar en archivo temporal para librosa
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_wav = tmp.name

    ap    = AudioProcessor()
    audio = ap.load_audio(tmp_wav)
    duration = len(audio) / ap.sample_rate

    st.info(f"Duración del audio cargado: **{duration:.1f} s**")

    max_duration = MAX_AUDIO_FRAMES / FPS   # 81.92 s
    truncated    = duration > max_duration
    if truncated:
        st.warning(
            f"⚠️ Audio truncado a los primeros **{max_duration:.0f} segundos** "
            f"(MAX_AUDIO_FRAMES = {MAX_AUDIO_FRAMES})"
        )

    # ── Visualización de entrada ───────────────────────────────────────────
    st.subheader("Visualización de entrada")
    col1, col2 = st.columns(2)

    with col1:
        fig_wf = _plot_waveform(audio, ap.sample_rate)
        st.pyplot(fig_wf)
        plt.close(fig_wf)

    mel = ap.compute_spectogram(audio)

    with col2:
        fig_sp = _plot_spectrogram(mel, duration)
        st.pyplot(fig_sp)
        plt.close(fig_sp)

    st.divider()

    # ── Opciones de decodificación ─────────────────────────────────────────
    st.subheader("Transcripción")
    decode_mode = st.radio(
        "Estrategia de decodificación:",
        ["Greedy", "Temperature sampling (T=0.8)"],
        horizontal=True,
    )

    transcribe_btn = st.button(
        "Transcribir",
        type="primary",
        disabled=(model is None),
    )

    if transcribe_btn:
        device = next(model.parameters()).device

        # Preparar tensor de audio
        mel_input = mel[:, :MAX_AUDIO_FRAMES]   # truncar si hace falta
        audio_tensor = (
            torch.tensor(mel_input, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        tokens = []
        midi_obj = None

        try:
            if decode_mode == "Greedy":
                with st.spinner("Generando transcripción (Greedy)..."):
                    raw = model.generate(
                        src_audio=audio_tensor,
                        start_token=midi_processor.token_sos,
                        max_len=1500,
                        end_token=midi_processor.token_eos,
                    )
                tokens = raw

            else:
                st.write("Generando con Temperature sampling (T=0.8)…")
                progress_bar = st.progress(0)
                tokens = _predict_sampling(
                    model,
                    audio_tensor,
                    midi_processor,
                    device,
                    temperature=0.8,
                    max_len=1500,
                    progress_bar=progress_bar,
                )
                progress_bar.progress(1.0)

            midi_obj = midi_processor.decode_midi(tokens)

        except Exception as exc:
            st.error(f"Error durante la transcripción: {exc}")

        if midi_obj is not None:
            notes = midi_obj.instruments[0].notes if midi_obj.instruments else []
            num_notes   = len(notes)
            est_duration = max((n.end for n in notes), default=0.0)

            # ── Resultado ─────────────────────────────────────────────────
            st.divider()
            st.subheader("Resultado")

            m1, m2 = st.columns(2)
            m1.metric("Notas detectadas", num_notes)
            m2.metric("Duración estimada", f"{est_duration:.1f} s")

            if notes:
                fig_pr = _plot_piano_roll(notes, est_duration)
                st.pyplot(fig_pr)
                plt.close(fig_pr)
            else:
                st.info("No se detectaron notas en la transcripción.")

            # Serializar MIDI a bytes
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_mid:
                midi_obj.write(tmp_mid.name)
                tmp_mid_path = tmp_mid.name

            with open(tmp_mid_path, "rb") as f:
                midi_bytes = f.read()
            os.unlink(tmp_mid_path)

            st.download_button(
                label="⬇️  Descargar MIDI",
                data=midi_bytes,
                file_name="transcripcion_magdalena.mid",
                mime="audio/midi",
            )

# ── Pie de página ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "TFG — María Piedad Castex Sierra — UCLM 2026"
    "</p>",
    unsafe_allow_html=True,
)
