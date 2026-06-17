EXP_NAME = "exp01_reduced_data"
EXP_DESCRIPTION = (
    "Eje 1: solo reducción del número de muestras de entrenamiento. "
    "Longitud de secuencia idéntica a v1_baseline. "
    "Arquitectura escalada (EMBED_DIM=512, 6 capas, NHEAD=8) para aprovechar A100 80 GB. "
    "Objetivo: validar que la arquitectura converge con más épocas por sesión de Colab."
)

# --- Datos ---
# Subconjunto fijo con semilla 42 para garantizar reproducibilidad entre sesiones
MAX_SAMPLES_TRAIN = 150
MAX_SAMPLES_VAL   = 30

# --- Longitud de secuencia (igual que v1_baseline) ---
# Límites explícitos para que no dependan de la constante global de maestro_dataset.py
MAX_AUDIO_FRAMES = 4096
MAX_MIDI_TOKENS  = 1500

# --- Hiperparámetros de entrenamiento ---
# Con A100 80 GB solo se usaban 3.3 GB con BATCH_SIZE=16 (4.1% de VRAM).
# BATCH_SIZE escalado a 64 (x4) para aprovechar la VRAM disponible.
# GRAD_ACCUMULATION_STEPS = 1: con batch real suficientemente grande no es necesaria la acumulación.
# Batch efectivo = 64.
BATCH_SIZE              = 64
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
CHECKPOINT_DIR = "checkpoints/exp01_reduced_data"
DRIVE_LOG_PATH = "/content/drive/MyDrive/TFG_Project/MPCS/checkpoints/exp01_reduced_data/"
