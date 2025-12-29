import librosa
import numpy as np
import torch
import torchaudio.transforms as T

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=229, n_fft=2048, hop_length=320, fmin=31, fmax=8000):
        """
        Clase para transformar audio crudo en Espectrogramas Mel Logarítmicos.
        
        Parámetros establecidos:
        - sample_rate: 16000 Hz 
        - n_fft: 2048 muestras (ventana de análisis de 128ms) 
        - hop_length: 0,02s * 16000 = 320 muestras (desplazamiento de 20ms) 
        - n_mels: 229   Estándar en registros de piano para cubrir 88 teclas, 
                        suficiente para que haya 1 o 2 bandas por semitono.
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def load_audio(self, file_path):
        """
        Carga del archivo de audio y remuestreo al sample_rate especificado.
        
        """
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    
    def compute_spectogram(self, audio):
        """
        Conversion del archivo de audio o ndarray a Espectrograma de Mel
        
        """
        # Si el audio no estaba cargado se carga
        if isinstance(audio, str):
            audio, _ = self.load_audio(audio)

        # Cálculo de espectrograma (escala lineal)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

        # Pasamos la amplitud a decibelios 
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=60)

        # Normalizamos para facilitar el trabajo a la red
        mel_normalized = (mel_spectrogram_db + 60)/60 

        # Devolvemos el espectrograma tal que:
        # Shape -> (n_mels, Time) -> (229, Time)
        return mel_normalized 
        
       

    
    def visualize(self, mel_spectrogram_db):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10,4))
        librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=self.sample_rate, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()