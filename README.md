# Magdalena v1.0 — Piano Transcription System

A deep-learning system that transcribes piano audio recordings into MIDI notation using a sparse Transformer encoder-decoder architecture trained on the MAESTRO dataset.

---

## What it does

Given a piano `.wav` recording, Magdalena outputs a `.mid` file containing the transcribed notes (pitch, onset, offset, velocity). The model is an encoder-decoder Transformer: a **Sparsifiner-based encoder** compresses the audio spectrogram, and a standard **autoregressive decoder** generates MIDI tokens one step at a time.

---

## Project structure

```
Magdalena/
├── train.py                        # Main training entry point
├── utils.py                        # Model config + loss logging helpers
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── maestro-v3.0.0_metadata.csv # Dataset split index
│   └── raw/maestro-v3.0.0/        # Audio (.wav) and MIDI pairs
│
├── src/
│   ├── data/
│   │   ├── audio_proc.py           # WAV → 229-band log-mel spectrogram
│   │   ├── midi_proc.py            # MIDI ↔ token vocabulary (vocab_size=311)
│   │   └── maestro_dataset.py      # PyTorch Dataset + DataLoaders
│   │
│   ├── models/
│   │   ├── transformer.py          # PianoTranscriptionModel (full model)
│   │   ├── encoder.py              # Standard AudioEncoder (reference, unused)
│   │   ├── sparsifiner.py          # Sparsifiner sparse-attention core
│   │   └── layers/
│   │       ├── sparsifiner_encoder.py  # AudioSparsifinerEncoder (active encoder)
│   │       ├── decoder.py              # MidiDecoder (autoregressive)
│   │       └── positional_encoding.py  # Sinusoidal positional encoding
│   │
│   ├── training/
│   │   └── trainer.py              # Trainer class (legacy, superseded by train.py)
│   ├── evaluate.py                 # Evaluation with mir_eval F1 metrics
│   └── tests/
│       ├── test_full_model.py
│       ├── test_encoder.py
│       ├── test_decoder.py
│       ├── test_dataset.py
│       ├── test_input.py
│       └── test_midi.py
│
└── checkpoints/                    # Auto-created during training
    ├── model/                      # model_epoch_N.pth
    └── optimizer/                  # opt_epoch_N.pth
```

---

## Architecture

### Input → Output

```
Piano .wav  →  Log-Mel Spectrogram (229 × T)
            →  AudioSparsifinerEncoder  →  memory (T × 256)
            →  MidiDecoder (autoregressive) →  token sequence
            →  MidiProcessor.decode_midi()  →  .mid file
```

### AudioSparsifinerEncoder

An adaptation of [Sparsifiner](https://arxiv.org/abs/2303.13755) (sparse instance-dependent attention for Vision Transformers) repurposed for variable-length audio sequences. Instead of patch embeddings, a linear projection maps each mel-frequency frame (229 dims) to the model dimension. Key properties:

- **Sparse attention mask**: a low-rank MaskPredictor selects the top-K most relevant frames per query (`attn_keep_rate=0.25` → 25% of frames attend to each other)
- **Dynamic slicing**: projection matrices are sliced to the actual sequence length at runtime, so sequences shorter than `max_seq_len` work without padding errors
- No token pruning is applied in the encoder (only attention sparsity)

### MidiDecoder

Standard `nn.TransformerDecoder` with:
- Causal (triangular) self-attention mask — teacher forcing during training, greedy/sampling during inference
- Cross-attention over the encoder memory
- Output linear layer projecting to the 311-token vocabulary

### Token vocabulary (311 tokens)

| Range | Type | Count |
|---|---|---|
| 0 | `<pad>` | 1 |
| 1 – 88 | Note-On (A0–C8) | 88 |
| 89 – 176 | Note-Off | 88 |
| 177 – 276 | Time-Shift (10 ms steps, up to 1 s per token) | 100 |
| 277 – 308 | Velocity (32 bins, 0–127) | 32 |
| 309 | `<sos>` | 1 |
| 310 | `<eos>` | 1 |

---

## Dataset

**MAESTRO v3.0.0** — paired piano audio/MIDI recordings from the International Piano-e-Competition.

- Download: [magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro)
- Place files under `data/raw/maestro-v3.0.0/` and the metadata CSV at `data/maestro-v3.0.0_metadata.csv`
- The CSV provides `train` / `validation` / `test` splits used by `MaestroDataset`

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (the training script uses mixed-precision AMP; CPU fallback exists but is impractical for 80 epochs)

---

## Installation

```bash
git clone <repo-url>
cd Magdalena
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Training

Edit the dataset paths at the top of `train.py`:

```python
csv_path='data/maestro-v3.0.0_metadata.csv',
root_dir='data/raw/maestro-v3.0.0/maestro-v3.0.0',
```

Then run:

```bash
python train.py
```

Key hyperparameters (top of `train.py`):

| Parameter | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | 2 | Effective batch = 16 via gradient accumulation |
| `GRAD_ACCUMULATION_STEPS` | 8 | |
| `LEARNING_RATE` | 1e-4 | AdamW |
| `EPOCHS` | 80 | |
| `embed_dim` | 256 | Increase to 512 if VRAM allows |
| `num_encoder_layers` | 4 | Sparsifiner blocks |
| `num_decoder_layers` | 4 | Transformer decoder blocks |
| `nhead` | 4 | Attention heads |

Checkpoints are saved after each epoch under `checkpoints/model/` and `checkpoints/optimizer/`. Training automatically resumes from the latest checkpoint if one is found.

### Evaluation

Edit `CSV_PATH`, `ROOT_DIR`, and `CHECKPOINT_PATH` in `src/evaluate.py`, then:

```bash
python src/evaluate.py
```

Reports average F1-scores on the validation split:

1. **Onset F1** — correct note starts (±50 ms tolerance)
2. **Onset + Offset F1** — correct note starts and ends
3. **Onset + Offset + Velocity F1** — full note accuracy including dynamics

### Running tests

```bash
python src/tests/test_midi.py
python src/tests/test_encoder.py
python src/tests/test_full_model.py
```

---

## Audio processing parameters

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 16 000 Hz | Standard for music ML |
| FFT window | 2048 samples (128 ms) | Frequency resolution |
| Hop length | 320 samples (20 ms) | 50 fps temporal resolution |
| Mel bands | 229 | ~1–2 bands per semitone across 88 piano keys |
| Frequency range | 31 – 8 000 Hz | Covers piano fundamentals |
| Normalization | (dB + 60) / 60 | Maps [−60, 0] dB → [0, 1] |

---

## Current state

**Complete**
- Full training pipeline with gradient accumulation, mixed precision (AMP), gradient clipping, and checkpoint resume
- MIDI tokenizer with full encode/decode round-trip
- Sparsifiner encoder adapted for variable-length audio
- Autoregressive decoder with teacher forcing and causal masking
- Evaluation script using mir_eval F1 metrics (onset / offset / velocity)
- Unit and integration tests for each component

**Work in progress / Known limitations**
- Dataset paths are hardcoded to Google Drive paths in `train.py` and `evaluate.py` — must be updated for local runs
- `MAX_AUDIO_FRAMES = 4096` (~82 s at 50 fps) and `MAX_MIDI_TOKENS = 1500` are hard truncation limits for long pieces
- `encoder.py` (standard `AudioEncoder`) is an unused baseline; the active encoder is `AudioSparsifinerEncoder`
- `src/training/trainer.py` is a legacy `Trainer` class superseded by the standalone `train.py` script
- Desktop GUI (`customtkinter`) and MusicXML export (`music21`) are listed in requirements but not yet implemented
- No inference chunking logic for audio longer than `max_seq_len`

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch` / `torchaudio` | Deep learning framework |
| `einops` | Tensor dimension manipulation in Sparsifiner |
| `librosa` | Log-mel spectrogram extraction |
| `pretty_midi` | MIDI parsing and synthesis |
| `mir_eval` | Standard music transcription evaluation metrics |
| `pandas` | MAESTRO metadata CSV handling |
| `tqdm` | Training progress bars |

See `requirements.txt` for the full version-pinned list.
