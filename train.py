# main.py o train.py
from src.data.audio_proc import AudioProcessor
from src.data.midi_proc import MidiProcessor
from src.data.maestro_dataset import get_dataloaders

# 1. Configuración (Aquí defines tus parámetros corregidos)
ap = AudioProcessor(fmax=8000, n_mels=229) 
mp = MidiProcessor() # Tus parámetros de MIDI

# 2. Obtener los loaders pasando los objetos 'ap' y 'mp'
train_loader, val_loader = get_dataloaders(
    csv_path='src/data/maestro-v3.0.0_metadata.csv',
    root_dir='src/data/raw/maestro-v3.0.0/maestro-v3.0.0',
    audio_processor=ap,  # <--- Inyección de dependencia
    midi_processor=mp,
    batch_size=4
)

# 3. Comprobar
for batch_audio, batch_midi in train_loader:
    print("Audio shape:", batch_audio.shape) # Debería ser (4, 229, Tiempo_Largo)
    print("Midi shape:", batch_midi.shape)   # Debería ser (4, Tiempo_Notas)
    break