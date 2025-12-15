import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from .audio_proc import AudioProcessor
from .midi_proc import MidiProcessor


class MaestroDataset(Dataset):
    def __init__(self, root_dir, split='train', sr = 16000, max_seq_len = 2048):
        """
        Dataset para el conjunto MAESTRO.
        
        Args:
            root_dir (str): Directorio raíz del dataset.
            split (str): División del dataset ('train', 'validation', 'test').
            sr (int): Frecuencia de muestreo para el audio.
            max_seq_len (int): Longitud máxima de la secuencia de tokens MIDI.
        """
        
        self.root_dir = root_dir
        self.split = split
        self.max_seq_len = max_seq_len

        # Inicializar los procesadores
        self.audio_processor = AudioProcessor(sample_rate=sr)
        self.midi_processor = MidiProcessor()

        # Cargar el archivo CSV con las anotaciones
        csv_path = os.path.join(root_dir, f'maestro-v3.0.0.csv')
        
        if not os.path.exists(csv_path):
            print(f"AVISO: No se encuentra {csv_path}. Asegúrate de haber descargado MAESTRO.")
            self.metadata = pd.DataFrame() # DataFrame vacío para evitar crashes al iniciar
        else:
            df = pd.read_csv(csv_path)
            # Filtramos por el split deseado (entrenamiento, validación o test)
            self.metadata = df[df['split'] == split].reset_index(drop=True)
            print(f"Dataset cargado ({split}): {len(self.metadata)} muestras encontradas.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_proc = AudioProcessor()
        midi_proc = MidiProcessor()
        # 1. Obtener rutas de archivos desde el CSV
        row = self.metadata.iloc[idx]
        
        # Las rutas en el CSV son relativas (ej: '2018/01.wav')
        audio_path = os.path.join(self.root_dir, row['audio_filename'])
        midi_path = os.path.join(self.root_dir, row['midi_filename'])

        # 2. Procesar AUDIO (Espectrograma)
        try:
            spectrogram = audio_proc.compute_spectogram(audio_path)
        except Exception as e:
            print(f"Error cargando audio {audio_path}: {e}")
            # Retornar tensor vacío en caso de error grave (o manejar según prefieras)
            spectrogram = torch.zeros(1, audio_proc.n_mels, 100)

        # 3. Procesar MIDI (Tokens)
        try:
            tokens = midi_proc.process_midi(midi_path)
            tokens_tensor = torch.from_numpy(tokens).long() # Convertir a tensor Long (Enteros)
        except Exception as e:
            print(f"Error cargando MIDI {midi_path}: {e}")
            tokens_tensor = torch.zeros(1).long()

        return {
            'audio': spectrogram,  # [1, n_mels, tiempo]
            'tokens': tokens_tensor, # [num_tokens]
            'label': row['canonical_title'] # Útil para debug
        }

def collate_fn(batch):
    """
    Función auxiliar para cuando usemos DataLoader.
    Como cada canción dura diferente, no podemos apilarlas (stack) directamente.
    Esta función maneja el padding o devuelve listas.
    """
    # Por ahora, simplemente devolvemos una lista de diccionarios
    # Más adelante, para el entrenamiento, aquí haremos el PadSequence
    return batch  