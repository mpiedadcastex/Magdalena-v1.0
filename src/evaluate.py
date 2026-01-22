import os
import torch
import numpy as np
import pretty_midi

import mir_eval.transcription
import mir_eval.transcription_velocity

from tqdm import tqdm
import pandas as pd

from src.data.audio_proc import AudioProcessor
from src.data.midi_proc import MidiProcessor
from src.models.transformer import PianoTranscriptionModel
from utils import get_model_config

# --- CONFIGURACIÓN ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Definición de rutas y checkpoints
EPOCHS_COMPLETED = 8
# Asegúrate de que esta ruta existe. Si falló antes, verifica la carpeta en Drive.
CHECKPOINT_PATH = f"checkpoints/model/model_epoch_{EPOCHS_COMPLETED}.pth"

CSV_PATH = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv'
ROOT_DIR = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0'
OUTPUT_DIR = "evaluation_results"

# 2. Configuración de Recorte (Para evitar OOM y reducir tiempo)
# RECOMENDACIÓN: Pon 30 para probar rápido. Pon 90 si quieres la evaluación real.
TEST_DURATION = 30  
FPS = 50            # 20ms hop_length = 50 FPS

# Tolerancias Métricas
ONSET_TOLERANCE = 0.05      # 50ms
OFFSET_RATIO = 0.2          # 20% duración
VELOCITY_TOLERANCE = 0.1    # 10% velocidad

def midi_to_intervals(midi_path, max_time=None):
    """
    Lee un archivo MIDI y extrae intervalos, pitches y velocities.
    CORRECCIÓN: Acepta max_time para filtrar notas fuera del rango de evaluación.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error {e} al leer el archivo MIDI: {midi_path}")
        return np.array([]), np.array([]), np.array([])
    
    intervals = []
    pitches = []
    velocities = []

    for instrument in pm.instruments:
        for note in instrument.notes:
            # Si definimos un corte y la nota empieza después, la ignoramos
            if max_time is not None and note.start > max_time:
                continue
            
            intervals.append([note.start, note.end])
            pitches.append(note.pitch)
            velocities.append(note.velocity)

    if not intervals:
        return np.array([]), np.array([]), np.array([])
        
    return (
        np.array(intervals),
        np.array(pitches),
        np.array(velocities)
    )

def predict_greedy(model, audio_tensor, midi_processor:MidiProcessor, max_len=None):
    """
    Generación nota a nota (Greedy).
    """
    model.eval()
    sos = midi_processor.token_sos
    eos = midi_processor.token_eos

    generated_sequence = torch.tensor([[sos]], dtype=torch.long).to(DEVICE)

    # Límite de seguridad: Si dura 30s, ponemos margen para ~1000 tokens (aprox 30 eventos/segundo)
    if max_len is None:
        max_len = int(TEST_DURATION * 35) 

    print(f"Generando secuencia (Máx tokens: {max_len})...")

    with torch.no_grad():
        # tqdm para ver progreso de la canción actual
        for i in tqdm(range(max_len), desc="Tokens"):
            logits = model(audio_tensor, generated_sequence, tgt_padding_mask=None)
            last_token_logits = logits[:, -1, :]
            predicted_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)

            if predicted_token.item() == eos:
                print(" <EOS> Fin de canción detectado.")
                break
            
            generated_sequence = torch.cat([generated_sequence, predicted_token], dim=1)
            
            # Limpieza de VRAM cada 50 pasos
            if i % 50 == 0:
                torch.cuda.empty_cache()

    return generated_sequence.squeeze().cpu().numpy()

def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- INICIANDO EVALUACIÓN (Recorte: {TEST_DURATION}s) ---")
    print(f"Modelo: {CHECKPOINT_PATH}")

    # Inicializar procesadores
    ap = AudioProcessor()
    mp = MidiProcessor()

    # Cargar CSV
    try:
        df = pd.read_csv(CSV_PATH)
        val_df = df[df['split'] == 'validation']
        print(f"Total de archivos en Validación: {len(val_df)}")
    except Exception as e:
        print(f"Error al cargar el CSV: {CSV_PATH}")
        return

    # Cargar Modelo
    cfg = get_model_config()
    model = PianoTranscriptionModel(
        midi_processor=mp,
        encoder_cfg=cfg,
        embed_dim=256,       
        num_encoder_layers=4,
        num_decoder_layers=4,
        nhead=4
    ).to(DEVICE)

    try:
        # Cargar pesos ignorando errores si hay pequeña discrepancia, 
        # pero idealmente debe coincidir todo.
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error CRÍTICO al cargar los pesos: {e}")
        return
    
    metrics = {'onset': [], 'onset_offset': [], 'velocity': []}
    
    count = 0
    limit = 5 # Prueba con 5 archivos primero

    for idx, row in val_df.iterrows():
        if limit and count >= limit: break

        audio_filename = os.path.join(ROOT_DIR, row['audio_filename'])
        midi_filename_gt = os.path.join(ROOT_DIR, row['midi_filename'])

        if not os.path.exists(audio_filename):
            print(f"Audio no encontrado: {audio_filename}")
            continue
        
        # --- TU PETICIÓN: IMPRIMIR NOMBRE DEL ARCHIVO ---
        print(f"\n[{count+1}/{limit}] Procesando: {row['audio_filename']}")
        # -----------------------------------------------

        try:
            # 1. Procesar Audio + Recorte
            mel = ap.compute_spectogram(audio_filename) 
            
            max_frames = int(TEST_DURATION * FPS)
            if mel.shape[1] > max_frames:
                mel = mel[:, :max_frames]

            audio_tensor = torch.tensor(mel).unsqueeze(0).to(DEVICE)

            # 2. Inferencia
            pred_tokens = predict_greedy(model, audio_tensor, mp)

            # 3. Decodificar
            pred_midi_path = os.path.join(OUTPUT_DIR, f"pred_{count}.mid")
            pred_midi_obj = mp.decode_midi(pred_tokens, output_path=pred_midi_path)

            # 4. Calcular Métricas (Usando MAX_TIME)
            ref_ints, ref_ps, ref_vels = midi_to_intervals(midi_filename_gt, max_time=TEST_DURATION)
            est_ints, est_ps, est_vels = midi_to_intervals(pred_midi_path, max_time=TEST_DURATION)

            # Si no hay notas en alguno de los dos, saltamos (o ponemos 0)
            if est_ints.size == 0 or ref_ints.size == 0:
                print("  -> Advertencia: No se detectaron notas en el intervalo recortado.")
                continue

            # Onset F1
            _, _, on_f1 = mir_eval.transcription.precision_recall_f1(
                ref_intervals=ref_ints, ref_pitches=ref_ps,
                est_intervals=est_ints, est_pitches=est_ps,
                onset_tolerance=ONSET_TOLERANCE
            )
            
            # Onset & Offset F1
            _, _, on_off_f1 = mir_eval.transcription.precision_recall_f1(
                ref_intervals=ref_ints, ref_pitches=ref_ps,
                est_intervals=est_ints, est_pitches=est_ps,
                onset_tolerance=ONSET_TOLERANCE,
                offset_ratio=OFFSET_RATIO
            )

            # Velocity F1
            _, _, vel_f1 = mir_eval.transcription_velocity.precision_recall_f1(
                ref_intervals=ref_ints, ref_pitches=ref_ps, ref_velocities=ref_vels,
                est_intervals=est_ints, est_pitches=est_ps, est_velocities=est_vels,
                onset_tolerance=ONSET_TOLERANCE,
                offset_ratio=OFFSET_RATIO,
                velocity_tolerance=VELOCITY_TOLERANCE
            )

            print(f"  -> Onset F1: {on_f1:.4f} | Onset+Off: {on_off_f1:.4f}")

            metrics['onset'].append(on_f1)
            metrics['onset_offset'].append(on_off_f1)
            metrics['velocity'].append(vel_f1)

            count += 1

        except Exception as e:
            print(f"Error en archivo {idx}: {e}")
            continue

    # --- RESULTADOS FINALES ---
    print("\n" + "="*50)
    print("RESULTADOS PROMEDIO (F1-Score)")
    print("="*50)
    if len(metrics['onset']) > 0:
        print(f"1. Onset:                  {np.mean(metrics['onset']):.4f}")
        print(f"2. Onset & Offset:         {np.mean(metrics['onset_offset']):.4f}")
        print(f"3. Onset, Offset & Velocity: {np.mean(metrics['velocity']):.4f}")
        print("-" * 50)
        print(f"Evaluados: {len(metrics['onset'])} archivos.")
    else:
        print("No se pudieron calcular métricas.")
    print("="*50)

if __name__ == "__main__":
    evaluate()