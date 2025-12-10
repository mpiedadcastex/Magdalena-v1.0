import librosa
import numpy as np
import torch
import torchaudio.transforms as T

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=229, n_fft=2048, hop_length=320):
        """
        Clase para transformar audio crudo en Espectrogramas Mel Logarítmicos.
        
        Parámetros establecidos:
        - sample_rate: 16000 Hz 
        - n_fft: 2048 muestras (ventana de análisis) 
        - hop_length: 320 muestras (20ms de desplazamiento) 
        - n_mels: 229 (Estándar en MAESTRO/Onsets&Frames para cubrir 88 teclas)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_audio(self, file_path):
        """Carga un archivo de audio y lo remuestrea al sample_rate especificado."""
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def compute_spectogram(self, audio):
        """
        Calcula el Log Mel Spectrogram.
        Retorna: Tensor de PyTorch de dimensiones [Canales, Frecuencia, Tiempo]
        """
        # 1. Cargar audio si es una ruta
        if isinstance(audio, str):
            audio = self.load_audio(audio)
        
        # 2. Calcular la Short-Time Fourier Transform (STFT)
        # Generamos la "imagen acústica" compleja
        stft = librosa.stft(
            audio, 
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            window = 'hann'
        )

        # Calculamos la magnitud (energía)
        spectrogram = np.abs(stft)**2

        # 3. Convertir a Mel Spectrogram
        # Creamos el banco de filtros Mel
        mel_basis = librosa.filters.mel(
            sr = self.sample_rate,
            n_fft = self.n_fft,
            n_mels = self.n_mels,
            fmin = 30, # Frecuencia mínima del piano
            fmax = 8000 # Frecuencia máxima del piano relevante
        )

        mel_spectrogram = np.dot(mel_basis, spectrogram)

        # 4. Aplicar la escala logarítmica
        # COnvertimos a decibelios (log) para comprimir el rango dinámico
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # 5. Convertir a Tensor de PyTorch y añadir dimensión de canal
        # Salida esperada: [Canales -> 1, Frecuencia, Tiempo]
        tensor = torch.from_numpy(log_mel_spectrogram).float()
        return tensor.unsqueeze(0)  # Aquí es donde añadimos la dimensión de canal
    
    def visualize(self, spectrogram_tensor):
        """Método auxiliar para visualizar el espectrograma usando matplotlib."""
        import matplotlib.pyplot as plt
        
        spec_np =spectrogram_tensor.squeeze(0).numpy()  # Eliminar dimensión de canal

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spec_np,
            sr = self.sample_rate,
            hop_length= self.hop_length,
            x_axis='time',
            y_axis='mel',
            fmin=30,
            fmax=8000
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log Mel Spectrogram ')
        plt.tight_layout()
        plt.show()