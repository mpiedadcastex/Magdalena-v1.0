import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from .audio_proc import AudioProcessor
from .midi_proc import MidiProcessor

# --- MEDIDA DE EMERGENCIA: CORTE DURO ---
MAX_AUDIO_FRAMES = 45000  # Límite de seguridad para Audio (~15 min)
MAX_MIDI_TOKENS = 10000

class MaestroDataset(Dataset):
    def __init__(self, csv_file, root_dir, audio_processor, midi_processor, split='train'):
        """
        Dataset para el conjunto MAESTRO.
        
        Args:
            csv_file (str): ruta al csv de los metadatos de MAESTRO
            root_dir (str): Directorio raíz del dataset
            audio_processor (AudioProcessor): Procesador de audio
            midi_processor (MidiProcessor): Procesador de midi
            split (str): 'train', 'validation' o 'test'
            
        """
        # Cargamos el csv y filtramos por el split seleccionado
        self.metadata = pd.read_csv(csv_file)
        self.metadata = self.metadata[self.metadata['split'] == split ]

        self.root_dir = root_dir

        # Inicializar los procesadores
        self.audio_processor = audio_processor
        self.midi_processor = midi_processor

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Obtenemos las rutas del audio y el midi del csv
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.root_dir, row['audio_filename'])
        midi_path = os.path.join(self.root_dir, row['midi_filename'])

        # Procesamos el audio con el AudioProcessor
        # Output -> numpy array (229, time)
        spectrogram = self.audio_processor.compute_spectogram(audio_path)

        # Procesamos el MIDI con el MidiProcessor
        # Output -> tokens
        midi_tokens = self.midi_processor.process_midi(midi_path)

        # Corte de seguridad a ver si se puede entrenar
        # AUDIO
        if (spectrogram.shape[1] > MAX_AUDIO_FRAMES):
            spectrogram = spectrogram[:, :MAX_AUDIO_FRAMES]
        
        if (len(midi_tokens) > MAX_MIDI_TOKENS):
            midi_tokens = midi_tokens[:MAX_MIDI_TOKENS]
        
        # Convertimos ambas salidas a tensores
        # Audio -> Float    Midi -> Long
        spectrogram_tensor = torch.from_numpy(spectrogram).float()
        midi_tensor = torch.tensor(midi_tokens).long()

        return spectrogram_tensor, midi_tensor
        
def collate_fn(batch):
    """
    Esta función se le pasa al DataLoader para que sepa cómo juntar 
    audios de distinta duración en un solo batch usando Padding.
    """
    # Batch -> (Spectrogram1, Midi1)
    # El * "desempaqueta" batch, separa los espectrogramas y los midis
    # zip los une y crea una lista independiente con cada uno de los tipos
    spectrograms, midis = zip(*batch)

    """PADDING DE AUDIO"""
    # Antes de hacer el padding debemos poner el tiempo en la primera posicion
    # ya que la funcion de padding trabaja en la primera dimension
    spectrograms_permuted = [s.permute(1, 0) for s in spectrograms]

    # Añadimos la dimension de batch y establecemos como valor de relleno el 0, la falta de informacion
    spectrograms_padded = pad_sequence(spectrograms_permuted, batch_first=True, padding_value=0.0)

    # Volvemos a la forma (Batch, 229, time)
    spectrograms_padded = spectrograms_padded.permute(0, 2, 1)

    """PADDING DE MIDI"""
    midis_padded = pad_sequence(midis, batch_first=True, padding_value=0)

    return spectrograms_padded, midis_padded
    

def get_dataloaders(csv_path, root_dir, audio_processor, midi_processor, batch_size=1):
    """
    Función para crear los DataLoaders de Train y Validation pasando los procesadores configurados
    """
    # Se crean los Datasets pasando las instancias
    train_ds = MaestroDataset(csv_path, root_dir, audio_processor, midi_processor, split='train')
    val_ds = MaestroDataset(csv_path, root_dir, audio_processor, midi_processor, split='validation')

    # Creamos los Loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=2
        #,pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
        #,pin_memory=True
    )

    return train_loader, val_loader