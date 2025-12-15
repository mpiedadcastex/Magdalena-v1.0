import os
import requests
import zipfile

# URL oficial del dataset MAESTRO v3.0.0
# Versión solo MIDI (50MB)
DATASET_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip" 
# Si necesitas el audio (100GB+), usa esta URL con precaución:
# DATASET_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"

def download_maestro(target_folder="data"):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    filename = DATASET_URL.split("/")[-1]
    filepath = os.path.join(target_folder, filename)

    print(f"Descargando {filename}...")
    response = requests.get(DATASET_URL, stream=True)
    
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print("Descarga completada. Descomprimiendo...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    
    print("¡Listo!")

if __name__ == "__main__":
    download_maestro()