import os
import torch
import numpy as np
import pretty_midi
import mir_eval
# Imports explícitos necesarios
import mir_eval.transcription
import mir_eval.transcription_velocity
from tqdm import tqdm
import pandas as pd
import traceback

from src.data.audio_proc import AudioProcessor
from src.data.midi_proc import MidiProcessor
from src.models.transformer import PianoTranscriptionModel
from utils import get_model_config

# --- CONFIGURACIÓN ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rutas
EPOCHS_COMPLETED = 36 
CHECKPOINT_PATH = f"checkpoints/model/model_epoch_{EPOCHS_COMPLETED}.pth"
CSV_PATH = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0_metadata.csv'
ROOT_DIR = '/content/drive/MyDrive/TFG_Data/maestro-v3.0.0/maestro-v3.0.0'
OUTPUT_DIR = "evaluation_results"

# Configuración de Recorte
TEST_DURATION = 30  # 30 segundos
FPS = 50            

# Tolerancias
ONSET_TOLERANCE = 0.05      
OFFSET_RATIO = 0.2          
VELOCITY_TOLERANCE = 0.1    

def midi_to_intervals(midi_path, max_time=None):
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
            if max_time is not None and note.start > max_time:
                continue
            intervals.append([note.start, note.end])
            pitches.append(note.pitch)
            velocities.append(note.velocity)

    if not intervals:
        return np.array([]), np.array([]), np.array([])
        
    return (np.array(intervals), np.array(pitches), np.array(velocities))

def predict_sampling(model, audio_tensor, midi_processor, max_len=None, temperature=1.0):
    """
    Genera notas usando muestreo probabilístico (rompe bucles repetitivos).
    temperature: 
      - 1.0 = Normal
      - < 1.0 (ej 0.8) = Más conservador (menos errores, más repetitivo)
      - > 1.0 (ej 1.2) = Más creativo (más variedad, más riesgo de error)
    """
    model.eval()
    sos = midi_processor.token_sos
    eos = midi_processor.token_eos
    
    generated_sequence = torch.tensor([[sos]], dtype=torch.long).to(DEVICE)
    if max_len is None: max_len = int(TEST_DURATION * 35)

    print(f"Generando con Sampling (T={temperature})...")

    with torch.no_grad():
        for i in tqdm(range(max_len)):
            logits = model(audio_tensor, generated_sequence, tgt_padding_mask=None)
            last_logits = logits[:, -1, :] / temperature  # Aplicar temperatura
            
            # Convertir logits a probabilidades
            probs = torch.softmax(last_logits, dim=-1)
            
            # Elegir el siguiente token basándose en la probabilidad (tira los dados)
            predicted_token = torch.multinomial(probs, num_samples=1)

            if predicted_token.item() == eos:
                break
            
            generated_sequence = torch.cat([generated_sequence, predicted_token], dim=1)
            if i % 50 == 0: torch.cuda.empty_cache()

    return generated_sequence.squeeze().cpu().numpy()

def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- INICIANDO EVALUACIÓN (Recorte: {TEST_DURATION}s) ---")
    
    ap = AudioProcessor()
    mp = MidiProcessor()

    try:
        df = pd.read_csv(CSV_PATH)
        val_df = df[df['split'] == 'validation']
    except Exception as e:
        print(f"Error CSV: {e}")
        return

    cfg = get_model_config()
    model = PianoTranscriptionModel(midi_processor=mp, encoder_cfg=cfg, embed_dim=256, num_encoder_layers=4, num_decoder_layers=4, nhead=4).to(DEVICE)

    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
        print(f"Modelo cargado.")
    except Exception as e:
        print(f"Error pesos: {e}")
        return
    
    metrics = {'onset': [], 'onset_offset': [], 'velocity': []}
    count = 0
    limit = 5 

    for idx, row in val_df.iterrows():
        if limit and count >= limit: break
        audio_filename = os.path.join(ROOT_DIR, row['audio_filename'])
        midi_filename_gt = os.path.join(ROOT_DIR, row['midi_filename'])

        if not os.path.exists(audio_filename): continue
        print(f"\n[{count+1}/{limit}] Procesando: {row['audio_filename']}")

        try:
            mel = ap.compute_spectogram(audio_filename) 
            max_frames = int(TEST_DURATION * FPS)
            if mel.shape[1] > max_frames: mel = mel[:, :max_frames]
            audio_tensor = torch.tensor(mel).unsqueeze(0).to(DEVICE)

            pred_tokens = predict_sampling(model, audio_tensor, mp)
            pred_midi_path = os.path.join(OUTPUT_DIR, f"pred_{count}.mid")
            mp.decode_midi(pred_tokens, output_path=pred_midi_path)

            ref_ints, ref_ps, ref_vels = midi_to_intervals(midi_filename_gt, max_time=TEST_DURATION)
            est_ints, est_ps, est_vels = midi_to_intervals(pred_midi_path, max_time=TEST_DURATION)

            if est_ints.size == 0 or ref_ints.size == 0:
                print("  -> Advertencia: No se detectaron notas.")
                continue

            # --- CORRECCIÓN DE NOMBRES DE FUNCIÓN Y UNPACKING ---
            
            # 1. Onset F1 (Usamos overlap pero ignoramos offset con offset_ratio=None si es posible, o usamos onset_precision...)
            # La forma más segura en mir_eval moderno:
            _, _, on_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_ints, ref_pitches=ref_ps,
                est_intervals=est_ints, est_pitches=est_ps,
                onset_tolerance=ONSET_TOLERANCE,
                offset_ratio=None 
            )
            
            # 2. Onset & Offset F1
            _, _, on_off_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_ints, ref_pitches=ref_ps,
                est_intervals=est_ints, est_pitches=est_ps,
                onset_tolerance=ONSET_TOLERANCE,
                offset_ratio=OFFSET_RATIO
            )

            # 3. Velocity F1 (Usamos el submódulo transcription_velocity)
            _, _, vel_f1, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
                ref_intervals=ref_ints, ref_pitches=ref_ps, ref_velocities=ref_vels,
                est_intervals=est_ints, est_pitches=est_ps, est_velocities=est_vels,
                onset_tolerance=ONSET_TOLERANCE,
                offset_ratio=OFFSET_RATIO,
                velocity_tolerance=VELOCITY_TOLERANCE
            )

            print(f"  -> Onset F1: {on_f1:.4f} | Onset+Off: {on_off_f1:.4f} | Vel: {vel_f1:.4f}")
            metrics['onset'].append(on_f1)
            metrics['onset_offset'].append(on_off_f1)
            metrics['velocity'].append(vel_f1)
            count += 1

        except Exception as e:
            traceback.print_exc()
            print(f"Error en archivo {idx}: {e}")
            continue

    print("\n" + "="*50)
    print("RESULTADOS PROMEDIO (F1-Score)")
    if len(metrics['onset']) > 0:
        print(f"1. Onset:                  {np.mean(metrics['onset']):.4f}")
        print(f"2. Onset & Offset:         {np.mean(metrics['onset_offset']):.4f}")
        print(f"3. Onset, Offset & Velocity: {np.mean(metrics['velocity']):.4f}")
    else:
        print("No se pudieron calcular métricas.")
    print("="*50)

if __name__ == "__main__":
    evaluate()