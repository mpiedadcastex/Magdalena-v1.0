import os
import torch
from src.data.dataset import MaestroDataset
from src.training.trainer import Trainer
from src.models.transformer import PianoTranscriptionModel

# --- 1. CONFIGURACIÓN DE COLAB Y RUTAS ---

# La ruta donde está la carpeta MAESTRO.
# Dirección en Google Drive para Colab (asegúrate de montar Drive primero)
DATA_ROOT = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0' 

# Direccion local (descomenta si trabajas localmente)
#DATA_ROOT = 'data/raw/maestro-v3.0.0/maestro-v3.0.0' # Manteniendo la estructura local por ahora

# Asegúrate de que las carpetas de salida existen
os.makedirs('checkpoints', exist_ok=True)


# --- 2. HIPERPARÁMETROS DEL MODELO Y ENTRENAMIENTO ---
config = {
    # ------------------ Modelo ------------------
    'n_mels': 229,           # Frecuencias Mel (de audio_proc.py)
    'vocab_size': 308,       # Tokens MIDI (de midi_proc.py + 1 para PAD, 307+1=308)
    'd_model': 512,          # Tamaño del Embedding y de la representación interna
    'nhead': 8,              # Número de cabezas de atención
    'num_encoder_layers': 4, # Cuatro bloques de Encoder
    'num_decoder_layers': 4, # Cuatro bloques de Decoder
    'dim_feedforward': 2048, # Dimensión interna de la FF
    'dropout': 0.1,
    
    # ------------------ Entrenamiento ---------------
    'learning_rate': 1e-4,   # Tasa de aprendizaje
    'weight_decay': 1e-2,    # Regularización L2
    'batch_size': 4,         # Tamaño del lote (ajustar a la memoria de la GPU de Colab)
    'num_epochs': 50,
    'num_workers': 2         # Para cargar datos en paralelo (ajustar en Colab)
}


# --- 3. INICIALIZACIÓN Y EJECUCIÓN ---

def main():
    print("--- Inicializando Transcriptor de Piano ---")
    
    # Asegúrate de que el vocab_size sea correcto (si reservaste el 0 para PAD)
    if config['vocab_size'] < 308:
        print("ADVERTENCIA: Vocabulario demasiado pequeño. Revisar.")

    # 1. Cargar Datos
    try:
        train_dataset = MaestroDataset(root_dir=DATA_ROOT, split='train')
        val_dataset = MaestroDataset(root_dir=DATA_ROOT, split='validation')
    except Exception as e:
        print(f"Error al cargar datasets: {e}. Asegúrate de que DATA_ROOT es correcta.")
        return

    # 2. Inicializar Modelo
    model = PianoTranscriptionModel(config)
    print(f"Modelo creado. Parámetros totales: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 3. Inicializar Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )

    # 4. Iniciar Entrenamiento
    print("\n>>>> INICIANDO BUCLE DE ENTRENAMIENTO <<<<")
    trainer.train(
        num_epochs=config['num_epochs'], 
        checkpoint_path='checkpoints/best_model.pt'
    )
    print(">>>> ENTRENAMIENTO FINALIZADO <<<<")

if __name__ == '__main__':
    main()