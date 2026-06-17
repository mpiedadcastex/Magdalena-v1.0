EXP_NAME = "exp02_short_clips"
EXP_DESCRIPTION = (
    "Eje 2: solo reducción de la longitud de los clips de audio y MIDI. "
    "Número de muestras idéntico a v1_baseline. "
    "Arquitectura escalada (EMBED_DIM=512, 6 capas, NHEAD=8) para aprovechar A100 80 GB. "
    "Objetivo: validar si clips más cortos aceleran el entrenamiento sin degradar la convergencia."
)

# --- Datos ---
MAX_SAMPLES_TRAIN = None
MAX_SAMPLES_VAL   = None

# --- Longitud de secuencia (ejes principales del experimento) ---
MAX_AUDIO_FRAMES = 2048   # baseline: 4096
MAX_MIDI_TOKENS  = 750    # baseline: 1500

# --- Hiperparámetros de entrenamiento ---
# Con A100 80 GB solo se usaban 3.3 GB con BATCH_SIZE=32 (4.1% de VRAM).
# BATCH_SIZE escalado a 128 (x4): las secuencias cortas (2048 frames vs 4096 de exp01)
# permiten el doble de batch que exp01 con la misma presión de memoria.
# GRAD_ACCUMULATION_STEPS = 1: sin acumulación, batch efectivo = 128.
BATCH_SIZE              = 128
GRAD_ACCUMULATION_STEPS = 1
LEARNING_RATE           = 1e-4
EPOCHS                  = 1000

# --- Arquitectura del modelo ---
# Escalada respecto a v1_baseline para aprovechar los 80 GB de VRAM del A100.
# EMBED_DIM: 256 → 512 (mayor capacidad representacional).
# Capas encoder/decoder: 4 → 6 (más profundidad).
# NHEAD: 4 → 8 (más cabezas de atención, alineado con EMBED_DIM=512).
EMBED_DIM          = 512
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NHEAD              = 8

# --- Rutas (Google Colab) ---
CSV_PATH  = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv'
ROOT_DIR  = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0'
CHECKPOINT_DIR = "checkpoints/exp02_short_clips"
DRIVE_LOG_PATH = "/content/drive/MyDrive/TFG_Project/MPCS/checkpoints/exp02_short_clips/"
