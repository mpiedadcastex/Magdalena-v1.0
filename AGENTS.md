# AGENTS.md - Guía Técnica para Agentes de Código

> **Proyecto**: Transcripción Automática de Música de Piano a Partitura usando Aprendizaje Profundo
> **Tipo**: Trabajo Fin de Grado (TFG)
> **Stack**: Python 3.10+ | PyTorch 2.1+ | Librosa | music21 | CustomTkinter

---

## 1. Visión General

Este proyecto implementa un sistema de **Automatic Music Transcription (AMT)** que convierte grabaciones de audio de piano en partituras musicales. El pipeline completo transforma señales de audio en representaciones simbólicas (MIDI) utilizando una arquitectura **Encoder-Decoder basada en Transformer** con mecanismos de **atención dispersa (Sparsifiner)** para manejar secuencias largas de forma eficiente.

**Datasets utilizados**:
- **MAESTRO v3.0.0** (Google Magenta): >200 horas de piano con alineación audio-MIDI
- **MAPS Dataset**: Piano sintético y real con alineación MIDI

---

## 2. Estructura del Proyecto

```
MPCS/
├── src/
│   ├── data/                  # Procesamiento de datos
│   │   ├── audio_proc.py      # Conversión audio → espectrograma Mel
│   │   ├── midi_proc.py       # Tokenización MIDI (encode/decode)
│   │   └── maestro_dataset.py # Dataset y DataLoaders para MAESTRO
│   ├── models/                # Arquitecturas neurales
│   │   ├── encoder.py         # Encoder Transformer básico (referencia)
│   │   ├── sparsifiner.py     # Implementación de atención dispersa
│   │   ├── transformer.py     # Modelo principal (PianoTranscriptionModel)
│   │   └── layers/
│   │       ├── sparsifiner_encoder.py  # Encoder con Sparsifiner
│   │       ├── decoder.py              # Decoder autorregresivo
│   │       └── positional_encoding.py  # Codificación posicional sinusoidal
│   ├── training/
│   │   └── trainer.py         # Clase Trainer (alternativa a train.py)
│   ├── tests/                 # Tests unitarios por componente
│   └── app/                   # GUI de escritorio (pendiente)
├── train.py                   # Script principal de entrenamiento
├── evaluate.py                # Evaluación con métricas mir_eval
├── download_data.py           # Descarga automática de MAESTRO
├── utils.py                   # Configuración del modelo Sparsifiner
├── config/                    # Archivos YAML (pendiente)
└── checkpoints/               # Pesos del modelo guardados
```

---

## 3. Comandos de Desarrollo

```bash
# Instalación de dependencias
pip install -r requirements.txt

# --- TESTS (ejecutar desde raíz del proyecto) ---
python test_train.py                    # Test bucle de entrenamiento con datos dummy
python test_sparsf_encoder.py           # Test encoder Sparsifiner
python src/tests/test_encoder.py        # Test encoder básico
python src/tests/test_decoder.py        # Test decoder MIDI
python src/tests/test_midi.py           # Test tokenización MIDI (encode/decode)
python src/tests/test_full_model.py     # Test end-to-end del modelo completo
python src/tests/test_dataset.py        # Test carga del dataset MAESTRO
python src/tests/test_input.py          # Test procesamiento de audio

# Ejecutar todos los tests
for f in src/tests/test_*.py; do python "$f"; done

# --- ENTRENAMIENTO (requiere GPU - diseñado para Google Colab) ---
python train.py

# --- EVALUACIÓN ---
python evaluate.py

# --- DATOS ---
python download_data.py                 # Descarga MAESTRO (solo MIDI, 50MB)
```

---

## 4. Pipeline de Datos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE COMPLETO                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Audio (.wav, 16kHz)                                                        │
│       │                                                                     │
│       │ AudioProcessor.compute_spectogram()                                 │
│       │ - STFT con ventana Hann (n_fft=2048, hop=320)                      │
│       │ - Banco de filtros Mel (n_mels=229, fmin=31Hz, fmax=8kHz)          │
│       │ - Conversión a dB + normalización [0,1]                            │
│       ▼                                                                     │
│  Mel Spectrogram  ──────────────────────────  Shape: (Batch, 229, Time)    │
│       │                                                                     │
│       │ AudioSparsifinerEncoder                                             │
│       │ - Proyección lineal (229 → embed_dim)                              │
│       │ - Positional Embedding                                              │
│       │ - N bloques Sparsifiner (Low-Rank Attention)                       │
│       ▼                                                                     │
│  Memory (contexto) ─────────────────────────  Shape: (Batch, Time, embed)  │
│       │                                                                     │
│       │ MidiDecoder (Transformer Decoder)                                   │
│       │ - Self-Attention con máscara causal                                │
│       │ - Cross-Attention con memory del encoder                           │
│       │ - Proyección a vocabulario (311 tokens)                            │
│       ▼                                                                     │
│  Logits ────────────────────────────────────  Shape: (Batch, Seq, 311)     │
│       │                                                                     │
│       │ Greedy Decoding / Beam Search                                       │
│       ▼                                                                     │
│  Tokens MIDI                                                                │
│       │                                                                     │
│       │ MidiProcessor.decode_midi()                                         │
│       ▼                                                                     │
│  Archivo MIDI (.mid)                                                        │
│       │                                                                     │
│       │ music21 (pendiente implementar)                                     │
│       │ - Cuantización rítmica                                              │
│       │ - Detección de compás                                               │
│       │ - Separación de voces                                               │
│       ▼                                                                     │
│  Partitura (MusicXML/PDF)                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Técnicas de Ingeniería Empleadas

### 5.1 Procesamiento de Audio

| Técnica | Implementación | Justificación |
|---------|----------------|---------------|
| **Espectrograma Mel Logarítmico** | `audio_proc.py:44-56` | Representa frecuencias en escala perceptual humana. La escala Mel aproxima la respuesta no lineal del oído. |
| **229 bandas Mel** | `audio_proc.py:7` | Cobertura completa de las 88 teclas del piano (A0=27.5Hz a C8=4186Hz) con 2-3 bandas por semitono. Valor estándar en AMT (Hawthorne et al., 2018). |
| **Sample rate 16kHz** | `audio_proc.py:7` | Suficiente para capturar armónicos del piano (fundamental máxima ~4kHz). Reduce cómputo vs 44.1kHz sin pérdida perceptual significativa. |
| **Hop length 320 (20ms)** | `audio_proc.py:7` | Resolución temporal de 50 frames/segundo. Balance entre precisión temporal y longitud de secuencia. |
| **Normalización [0,1]** | `audio_proc.py:59` | `(dB + 60) / 60` tras conversión a decibelios con `top_db=60`. Facilita convergencia de la red. |

### 5.2 Arquitectura del Modelo

| Técnica | Implementación | Justificación |
|---------|----------------|---------------|
| **Sparsifiner (Atención Dispersa)** | `sparsifiner.py` | Reduce complejidad de O(N²) a O(N·k) donde k << N. Crítico para secuencias de audio largas (>10,000 frames). |
| **Low-Rank Attention Approximation** | `sparsifiner.py:159-214` | Proyecta Q, K a dimensiones reducidas (`reduce_c_factor`, `reduce_n_factor`) para calcular "cheap attention" que guía la selección de posiciones importantes. |
| **Top-K Attention Masking** | `sparsifiner.py:268-277` | Selecciona solo las k posiciones más relevantes por fila de atención, forzando dispersión. `attn_keep_rate=0.25` significa 75% de ahorro. |
| **Positional Encoding Sinusoidal** | `positional_encoding.py` | Inyecta información posicional sin parámetros aprendibles. Permite generalización a longitudes no vistas. (Vaswani et al., 2017) |
| **Decoder Autorregresivo** | `decoder.py` | Genera tokens secuencialmente con máscara causal. Cross-attention permite "mirar" todo el audio codificado. |
| **DropPath (Stochastic Depth)** | `sparsifiner.py:82-104` | Regularización que descarta capas completas aleatoriamente durante entrenamiento. Mejora generalización. (Huang et al., 2016) |
| **Truncated Normal Initialization** | `sparsifiner.py:71-75` | Inicialización de pesos estándar para Transformers. Evita valores extremos que desestabilicen el entrenamiento inicial. |

### 5.3 Optimización del Entrenamiento

| Técnica | Implementación | Justificación |
|---------|----------------|---------------|
| **Teacher Forcing** | `train.py:135-143` | Durante entrenamiento, el decoder recibe los tokens correctos (no sus predicciones). Acelera convergencia significativamente. (Williams & Zipser, 1989) |
| **Gradient Accumulation** | `train.py:21,167-188` | Simula batch size mayor (efectivo=16) con VRAM limitada (batch real=2). Acumula gradientes durante 8 pasos antes de actualizar. |
| **Mixed Precision (AMP)** | `train.py:120,152` | Usa FP16 para forward/backward y FP32 para actualización de pesos. Reduce memoria ~50% y acelera cómputo en GPUs modernas. (Micikevicius et al., 2018) |
| **Gradient Clipping** | `train.py:181` | `clip_grad_norm_(max_norm=1.0)` previene explosión de gradientes, común en Transformers con secuencias largas. (Pascanu et al., 2013) |
| **AdamW Optimizer** | `train.py:68` | Adam con weight decay desacoplado. Estándar para Transformers. (Loshchilov & Hutter, 2019) |
| **Learning Rate 1e-4** | `train.py:23` | Valor estándar para fine-tuning de Transformers. Suficientemente pequeño para estabilidad, grande para progreso. |
| **CrossEntropyLoss con ignore_index** | `train.py:71` | `ignore_index=0` excluye tokens de padding del cálculo de pérdida. Crítico para secuencias de longitud variable. |
| **Checkpoint Resumption** | `train.py:73-115` | Sistema de reanudación automática que detecta el último checkpoint guardado y continúa desde esa época. |

### 5.4 Tokenización MIDI

| Técnica | Implementación | Justificación |
|---------|----------------|---------------|
| **Event-Based Tokenization** | `midi_proc.py` | Representa MIDI como secuencia de eventos discretos, no como piano roll. Más eficiente para secuencias largas. (Oore et al., 2020) |
| **Vocabulario de 311 tokens** | `midi_proc.py:51` | 88 Note-On + 88 Note-Off + 100 Time-Shift + 32 Velocity + 3 especiales. Balance entre expresividad y tamaño de vocabulario. |
| **Time-Shift en pasos de 10ms** | `midi_proc.py:18` | Resolución temporal suficiente para música de piano. Hasta 1 segundo por token (100 bins). |
| **Velocity cuantizada a 32 niveles** | `midi_proc.py:23` | Reduce vocabulario manteniendo expresividad dinámica perceptualmente equivalente. |
| **Ordenación Off-antes-de-On** | `midi_proc.py:112` | Cuando dos eventos coinciden en tiempo, Note-Off va primero. Evita ambigüedad en notas consecutivas del mismo pitch. |

---

## 6. Problemas Resueltos

### 6.1 Memoria GPU (Out of Memory)

| Problema | Causa | Solución | Ubicación |
|----------|-------|----------|-----------|
| OOM en forward pass | Secuencias de audio >4096 frames | Corte duro: `MAX_AUDIO_FRAMES=4096`, `MAX_MIDI_TOKENS=1500` | `maestro_dataset.py:12-13` |
| OOM con batch size > 2 | VRAM insuficiente en Colab (16GB) | Gradient Accumulation: batch=2, accumulation=8 → efectivo=16 | `train.py:20-21` |
| Fragmentación de memoria CUDA | Allocaciones/deallocaciones frecuentes | `PYTORCH_ALLOC_CONF="expandable_segments:True"` | `train.py:4` |
| Memoria creciente durante época | Tensores intermedios no liberados | `torch.cuda.empty_cache()` tras cada actualización | `train.py:189` |

### 6.2 Dimensiones y Shapes

| Problema | Causa | Solución | Ubicación |
|----------|-------|----------|-----------|
| Error en Sparsifiner con audio corto | Matriz de proyección de tamaño fijo vs. entrada variable | Slicing dinámico: `proj_n[:N, :]` | `sparsifiner.py:209-214` |
| TopK falla si k > N | Presupuesto de atención fijo mayor que secuencia | `real_k = min(self.attn_budget, N)` | `sparsifiner.py:271` |
| Incompatibilidad batch en padding | Secuencias de distinta longitud en batch | Custom `collate_fn` con `pad_sequence` | `maestro_dataset.py:70-95` |
| Máscara de padding incorrecta | Decoder atendía a tokens de relleno | `tgt_padding_mask = (decoder_input == 0)` | `train.py:147` |

### 6.3 Dependencias y Compatibilidad

| Problema | Causa | Solución | Ubicación |
|----------|-------|----------|-----------|
| `timm.models.layers` deprecado | Cambio de API en versiones recientes de timm | Funciones copiadas manualmente: `trunc_normal_`, `DropPath`, `to_2tuple` | `sparsifiner.py:28-104` |
| Rutas hardcodeadas para Colab | Desarrollo en Colab, ejecución local diferente | Variables al inicio de scripts (pendiente: config YAML) | `train.py:45-46` |

### 6.4 Entrenamiento

| Problema | Causa | Solución | Ubicación |
|----------|-------|----------|-----------|
| Gradientes NaN/Inf | Overflow en FP16 o división por cero | GradScaler + `torch.nan_to_num(attn)` | `train.py:120`, `sparsifiner.py:392` |
| Pérdida no decrece | Modelo aprendiendo del padding | `ignore_index=0` en CrossEntropyLoss | `train.py:71` |
| Entrenamiento interrumpido | Desconexiones de Colab | Sistema de checkpoints con reanudación automática | `train.py:73-115` |

---

## 7. Decisiones de Diseño

### ¿Por qué Sparsifiner y no atención estándar?
La atención estándar tiene complejidad O(N²) en memoria y cómputo. Una pieza de piano de 5 minutos a 50 fps genera ~15,000 frames. La matriz de atención sería de 15,000 × 15,000 = 225M elementos por cabeza. Sparsifiner reduce esto a O(N·k) donde k es el presupuesto de atención (típicamente 25% de N), permitiendo procesar secuencias largas en GPUs con memoria limitada.

### ¿Por qué 229 bandas Mel?
El piano tiene 88 teclas cubriendo frecuencias de 27.5Hz (A0) a 4186Hz (C8). Con 229 bandas Mel en el rango 31-8000Hz, cada semitono tiene aproximadamente 2-3 bandas dedicadas, suficiente para distinguir notas cercanas. Este valor es estándar en la literatura de AMT (Hawthorne et al., 2018; Kong et al., 2021).

### ¿Por qué Teacher Forcing sin Scheduled Sampling?
Teacher Forcing acelera significativamente la convergencia inicial. Scheduled Sampling (mezclar predicciones propias gradualmente) puede mejorar la robustez en inferencia pero complica el entrenamiento. Decisión: empezar con Teacher Forcing puro, evaluar, y añadir Scheduled Sampling si hay discrepancia train/inference significativa.

### ¿Por qué tokenización por eventos y no piano roll?
Un piano roll de 5 minutos a 50fps con 88 teclas tiene 15,000 × 88 = 1.32M celdas. La tokenización por eventos representa la misma información en ~5,000-10,000 tokens típicamente, mucho más eficiente para modelos autorregresivos.

### ¿Por qué no usar offset relativo en Time-Shift?
La implementación actual usa time-shifts absolutos desde el último evento. Alternativa: offsets relativos al onset anterior. Decisión: absolutos son más simples de implementar y debuggear; la diferencia en rendimiento es marginal según literatura.

---

## 8. Convenciones de Código

### Organización de Imports
```python
# 1. Biblioteca estándar
import os
import math
from types import SimpleNamespace

# 2. Terceros (PyTorch primero, luego alfabético)
import torch
import torch.nn as nn
import numpy as np
import librosa
import pretty_midi

# 3. Módulos locales (rutas desde src/)
from src.data.audio_proc import AudioProcessor
from src.models.transformer import PianoTranscriptionModel
```

### Nomenclatura
| Elemento | Convención | Ejemplo |
|----------|------------|---------|
| Clases | PascalCase | `AudioProcessor`, `MidiDecoder` |
| Funciones/métodos | snake_case | `compute_spectogram`, `encode_midi` |
| Constantes | UPPER_SNAKE | `BATCH_SIZE`, `MAX_AUDIO_FRAMES` |
| Variables | snake_case | `mel_spectrogram`, `token_mask` |

### Documentación de Shapes (OBLIGATORIO)
```python
# SIEMPRE documentar shapes en comentarios
x = x.permute(0, 2, 1)  # (Batch, 229, Time) -> (Batch, Time, 229)
logits = model(audio)   # Output: (Batch, Seq_Len, vocab_size)
```

### Docstrings (Español, formato Google)
```python
def metodo(self, param1: torch.Tensor, param2: int) -> torch.Tensor:
    """
    Descripción breve del método.
    
    Args:
        param1: Tensor de entrada con shape (Batch, Seq, Dim)
        param2: Número de capas a aplicar
        
    Returns:
        Tensor procesado con shape (Batch, Seq, Dim)
    """
```

---

## 9. Referencias Bibliográficas

### Arquitectura y Atención
- **Vaswani, A., et al. (2017)**. "Attention Is All You Need". *NeurIPS*. [Transformer original]
- **Wei, C., et al. (2023)**. "Sparsifiner: Learning Sparse Instance-Dependent Attention for Efficient Vision Transformers". *CVPR*. [Atención dispersa]
- **Kitaev, N., et al. (2020)**. "Reformer: The Efficient Transformer". *ICLR*. [Atención eficiente]

### Transcripción Musical (AMT)
- **Hawthorne, C., et al. (2018)**. "Onsets and Frames: Dual-Objective Piano Transcription". *ISMIR*. [Baseline AMT piano, 229 Mel bins]
- **Kong, Q., et al. (2021)**. "High-Resolution Piano Transcription with Pedals by Regressing Onset and Offset Times". *TASLP*. [Estado del arte AMT]
- **Hawthorne, C., et al. (2019)**. "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset". *ICLR*. [Dataset MAESTRO]

### Tokenización y Representación MIDI
- **Oore, S., et al. (2020)**. "This Time with Feeling: Learning Expressive Musical Performance". *Neural Computing and Applications*. [Event-based MIDI tokenization]
- **Huang, C.Z.A., et al. (2018)**. "Music Transformer". *ICLR*. [Transformers para música]

### Optimización de Entrenamiento
- **Micikevicius, P., et al. (2018)**. "Mixed Precision Training". *ICLR*. [AMP/FP16]
- **Loshchilov, I. & Hutter, F. (2019)**. "Decoupled Weight Decay Regularization". *ICLR*. [AdamW]
- **Pascanu, R., et al. (2013)**. "On the Difficulty of Training Recurrent Neural Networks". *ICML*. [Gradient clipping]
- **Huang, G., et al. (2016)**. "Deep Networks with Stochastic Depth". *ECCV*. [DropPath]

### Teacher Forcing
- **Williams, R.J. & Zipser, D. (1989)**. "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks". *Neural Computation*. [Teacher Forcing original]

---

## 10. Vocabulario MIDI

| Rango | Tipo | Cantidad | Descripción |
|-------|------|----------|-------------|
| 0 | PAD | 1 | Token de padding |
| 1-88 | Note-On | 88 | Inicio de nota (teclas MIDI 21-108) |
| 89-176 | Note-Off | 88 | Fin de nota |
| 177-276 | Time-Shift | 100 | Avance temporal (10ms - 1000ms) |
| 277-308 | Velocity | 32 | Dinámica (32 niveles cuantizados) |
| 309 | SOS | 1 | Start of Sequence |
| 310 | EOS | 1 | End of Sequence |
| **Total** | | **311** | |

---

## 11. Tests

Los tests son scripts independientes que validan cada componente:
- Usan datos sintéticos (tensores aleatorios) para no depender del dataset
- Verifican shapes de entrada/salida esperados
- Imprimen ✅/❌ según resultado

```bash
# Patrón para crear nuevos tests
python src/tests/test_<componente>.py
```

Seguir estructura de `src/tests/test_encoder.py` como referencia.

---

## 12. Entorno de Ejecución

### Google Colab (Principal)
Las rutas en `train.py` y `evaluate.py` están configuradas para Colab:
```python
csv_path = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/...'
root_dir = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/...'
```

### Configuración de GPU
```python
# Verificar disponibilidad
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evitar fragmentación de memoria
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

### Hiperparámetros por Defecto
| Parámetro | Valor | Notas |
|-----------|-------|-------|
| `embed_dim` | 256 | Subir a 512 con más VRAM |
| `num_encoder_layers` | 4 | |
| `num_decoder_layers` | 4 | |
| `nhead` | 4 | |
| `batch_size` | 2 | Limitado por VRAM |
| `grad_accumulation` | 8 | Batch efectivo = 16 |
| `learning_rate` | 1e-4 | |
| `epochs` | 80 | |
