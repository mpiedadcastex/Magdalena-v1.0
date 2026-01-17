import os
import torch
import numpy as np
import pretty_midi
import mir_eval
from tqdm import tqdm
import pandas as pd

from src.data.audio_proc import AudioProcessor
from src.data.midi_proc import MidiProcessor
from src.data.maestro_dataset import get_dataloaders
from src.models.transformer import PianoTranscriptionModel
from utils import get_model_config

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS_COMPLETED = 10
CHECKPOINT_PATH = "checkpoints/model/model_epoch_{EPOCHS_COMPLETED}.pth"    # Ajusta a la última versión del entrenamiento
CSV_PATH = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv'
ROOT_DIR = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0'
OUTPUT_DIR = "evaluation_results"

# Tolerancias
ONSET_TOLERANCE = 0.05      # 50ms
OFFSET_RATIO = 0.2          # 20% de la duración de la nota
VELOCITY_TOLERANCE = 0.1    # Margen de error en velocidad (normalizado escala 0-1)

def midi_to_intervals(midi_path):
    """
    Lee un archivo MIDI y extrae:
        - Intervalos
        - Pitches
        - Velocities
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error {e} al leer el archivo MIDI: {midi_path}")
        return None, None, None
    
    # Inicializamos las listas en las que guardaremos los resultados
    intervals = []
    pitches = []
    velocities = []

    for instrument in pm.instruments:

        # Actualizamos las listas con los valores de las notas 
        for note in instrument.notes:
            intervals.append([note.start, note.end])
            pitches.append(note.pitch)
            velocities.append(note.velocity)

        # Comprobación: Si no hay notas devolvemos los objetos vacios y lanzamos un mensaje
        if not intervals:
            print(f"No se detectaron notas")
            return np.array([]), np.array([]), np.array([])
        
        # Devolvemos los valores
        return (
            np.array(intervals),
            np.array(pitches),
            np.array(velocities)
        )
    
def calculate_metrics(ref_midi_path, est_midi_path):
    """
    Calculamos las tres métricas que hemos establecido:
        1º - Onset F1
        2º - Onset y Offset F1
        3º - Onset, Offset y Velocity F1
    """
    # Carga de datos
    ref_intervals, ref_pitches, ref_velocities = midi_to_intervals(ref_midi_path)
    est_intervals, est_pitches, est_velocities = midi_to_intervals(est_midi_path)

    if est_intervals.size == 0:
        return 0.0, 0.0, 0.0    # Si el modelo no ha predicho nada, todo a 0
    
    # 1º - Onset F1-Score
    # En este caso, unicamente nos importa acertar el tiempo de inicio de la nota (con la tolerancia establecida)
    # En matched_onset vamos a devolver los índices de las notas que coinciden
    matched_onset = mir_eval.transcription.match_notes(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=None # No aplica en este caso
    )

    # Calculamos P (Precision), R (Recall) y F1
    onset_p, onset_r, onset_f1 = mir_eval.transcription.precision_recall_f1(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=None
    )

    # 1º - Onset & Offset F1-Score
    # En este caso, nos importa acertar tanto el tiempo de inicio como de final de la nota
    # En matched_on_off vamos a devolver los índices de las notas que coinciden
    matched_on_off = mir_eval.transcription.match_notes(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=OFFSET_RATIO # Ahora sí que importa
    )

    on_off_p, on_off_r, on_off_f1 = mir_eval.transcription.precision_recall_f1(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=OFFSET_RATIO
    )

    # 1º - Onset & Offset F1-Score
    # Requiere acertar el tiempo de inicio y de final de la nota además de la intensidad con al que se toca
    vel_p, vel_r, vel_f1 = mir_eval.transcription_velocity.precision_recall_f1(
        ref_intervals, ref_pitches, ref_velocities,
        est_intervals, est_pitches, est_velocities,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=OFFSET_RATIO,
        velocity_tolerance=VELOCITY_TOLERANCE
    )

    return onset_f1,matched_onset, on_off_f1, matched_on_off, vel_f1