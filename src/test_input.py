import os
from data.audio_proc import AudioProcessor

# Rutas
processor = AudioProcessor()

# Busca un archivo de audio de prueba
try:
    sample_file = "data/raw/maestro-v3.0.0/maestro-v3.0.0/2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_03_Track03_wav.wav"

    print(f"Cargando archivo de audio de prueba: {sample_file}")
    print("Procesando...")
    spectrogram = processor.compute_spectogram(sample_file)

    print(f"\n✅ ÉXITO")
    print(f"Forma del tensor resultante: {spectrogram.shape}")
    print("Debería ser [1, 229 -> Frecuencias Mel, tiempo]")
    
    # Visualizar
    print("Generando imagen...")
    processor.visualize(spectrogram)

except Exception as e:
    print(f"❌ ERROR: {e}")