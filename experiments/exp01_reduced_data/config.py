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

# --- Hiperparámetros de entrenamiento (igual que v1_baseline) ---
BATCH_SIZE              = 8
GRAD_ACCUMULATION_STEPS = 4   # Batch efectivo = 32 (optimizado para GPU A10 24 GB)
LEARNING_RATE           = 1e-4
EPOCHS                  = 1000

# --- Arquitectura del modelo (igual que v1_baseline) ---
EMBED_DIM          = 256
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
NHEAD              = 4

# --- Rutas (Google Colab) ---
CSV_PATH  = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv'
ROOT_DIR  = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0'
CHECKPOINT_DIR = "checkpoints/exp01_reduced_data"
DRIVE_LOG_PATH = "/content/drive/MyDrive/TFG_Project/MPCS/checkpoints/exp01_reduced_data/"
