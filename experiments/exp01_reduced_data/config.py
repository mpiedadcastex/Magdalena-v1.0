EXP_NAME = "exp01_reduced_data"
EXP_DESCRIPTION = (
    "Eje 1: solo reducción del número de muestras de entrenamiento. "
    "Modelo y longitud de secuencia idénticos a v1_baseline. "
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
# BATCH_SIZE = 32 para A100 80 GB VRAM. Con BATCH_SIZE=64 se producía OOM (la atención
# es O(batch × seq²) con secuencias de 4096 frames). A 32 se mantiene margen suficiente.
# GRAD_ACCUMULATION_STEPS = 1: sin acumulación, batch efectivo = 32.
BATCH_SIZE              = 32
GRAD_ACCUMULATION_STEPS = 1
LEARNING_RATE           = 1e-4
EPOCHS                  = 1000

# --- Arquitectura del modelo (igual que v1_baseline) ---
# Mantenida intacta para compatibilidad con checkpoints existentes.
EMBED_DIM          = 256
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
NHEAD              = 4

# --- Rutas (Google Colab) ---
CSV_PATH  = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv'
ROOT_DIR  = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0'
CHECKPOINT_DIR = "checkpoints/exp01_reduced_data"
DRIVE_LOG_PATH = "/content/drive/MyDrive/TFG_Project/MPCS/checkpoints/exp01_reduced_data/"
