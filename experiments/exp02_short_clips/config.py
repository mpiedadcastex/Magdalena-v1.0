EXP_NAME = "exp02_short_clips"
EXP_DESCRIPTION = (
    "Eje 2: solo reducción de la longitud de los clips de audio y MIDI. "
    "Número de muestras y arquitectura idénticos a v1_baseline. "
    "Objetivo: validar si clips más cortos aceleran el entrenamiento sin degradar la convergencia."
)

# --- Datos ---
MAX_SAMPLES_TRAIN = None
MAX_SAMPLES_VAL   = None

# --- Longitud de secuencia (ejes principales del experimento) ---
MAX_AUDIO_FRAMES = 2048   # baseline: 4096
MAX_MIDI_TOKENS  = 750    # baseline: 1500

# --- Hiperparámetros de entrenamiento ---
# BATCH_SIZE = 32 para mantener coherencia de batch efectivo con exp01.
# Las secuencias cortas (2048 frames) permitirían un batch mayor, pero se unifica
# en 32 para que los experimentos sean comparables entre sí.
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
CHECKPOINT_DIR = "checkpoints/exp02_short_clips"
DRIVE_LOG_PATH = "/content/drive/MyDrive/TFG_Project/MPCS/checkpoints/exp02_short_clips/"
