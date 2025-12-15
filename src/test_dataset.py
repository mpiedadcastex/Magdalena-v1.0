import os
import torch
from data.dataset import MaestroDataset

DATA_ROOT = r"data\raw\maestro-v3.0.0\maestro-v3.0.0"

print(f"--- Probando Dataset en: {DATA_ROOT} ---")

# 1. Instanciar Dataset
# Intentamos cargar el split de 'train'
dataset = MaestroDataset(root_dir=DATA_ROOT, split='train')

# 2. Verificar longitud
print(f"Tamaño del dataset: {len(dataset)}")

if len(dataset) > 0:
    # 3. Cargar la primera muestra
    print("\nCargando primera muestra...")
    sample = dataset[0]
    
    spec = sample['audio']
    tokens = sample['tokens']
    title = sample['label']
    
    print(f"🎵 Título: {title}")
    print(f"📊 Forma del Espectrograma: {spec.shape} (Canal, Mel, Tiempo)")
    print(f"📝 Tokens MIDI (primeros 10): {tokens[:10]}")
    print(f"🔢 Total tokens: {tokens.shape[0]}")
    
    # Verificación de integridad
    if spec.shape[1] == 229:
        print("✅ Espectrograma correcto (229 Mels).")
    else:
        print("❌ Error en dimensiones del espectrograma.")
        
    if tokens.dtype == torch.int64:
        print("✅ Tokens en formato correcto (LongTensor).")
else:
    print("⚠️ No se encontraron datos en el CSV o no existe el CSV.")
    print("Asegúrate de que 'data/raw' contiene la carpeta descomprimida de MAESTRO y el archivo .csv")