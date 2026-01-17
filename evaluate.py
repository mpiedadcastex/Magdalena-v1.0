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
CHECKPOINT_PATH = f"/content/drive/MyDrive/TFG_Project/MPCS/checkpoints/model/model_epoch_{EPOCHS_COMPLETED}.pth"    # Ajusta a la última versión del entrenamiento
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

    # Calculamos P (Precision), R (Recall) y F1
    onset_p, onset_r, onset_f1 = mir_eval.transcription.precision_recall_f1(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=None
    )

    # 1º - Onset & Offset F1-Score
    # En este caso, nos importa acertar tanto el tiempo de inicio como de final de la nota
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

    return onset_f1, on_off_f1, vel_f1


def predict_greedy(model, audio_tensor, midi_processor:MidiProcessor, max_len=2048):
    """
    Generación nota a nota (Greedy)
    """
    model.eval()

    # Establecemos los tokens especiales de inicio y fin con los atributos del procesador midi
    sos = midi_processor.token_sos
    eos = midi_processor.token_eos

    # Iniciamos la secuencia con <SOS>
    generated_sequence = torch.tensor([[sos]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        for _ in range(max_len):
            # Forward pass
            # Ponemos tgt_padding_mask a None ya que no es necesaria con un batch size = 1
            logits = model(audio_tensor, generated_sequence, tgt_padding_mask=None)

            # Obtener última predicción
            last_token_logits = logits[:, -1, :]
            predicted_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)

            # Comprobamos si es el fin de la secuencia
            if predicted_token.item() == eos:
                break
            
            # Concatenamos en la secuencia
            generated_sequence = torch.cat([generated_sequence, predicted_token], dim=1)

    return generated_sequence.squeeze().cpu().numpy()


def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- INICIANDO EVALUACIÓN ---")
    print(f"Modelo: {CHECKPOINT_PATH}")

    # 1 - Inicializamos los procesadores
    ap= AudioProcessor()
    mp = MidiProcessor()

    # 2 - Cargamos el split de validación del dataset
    try:
        df = pd.read_csv(CSV_PATH)
        val_df = df[df['split'] == 'validation']
        print(f"Total de archivos en Validación: {len(val_df)}")
    except Exception as e:
        print(f"Error al cargar el CSV: {CSV_PATH}")

    # 3 - Cargamos el Modelo
    cfg = get_model_config()

    # Copiamos los hiperparámetros que utilizamos en el train del modelo
    model = PianoTranscriptionModel(
        midi_processor=mp,
        encoder_cfg=cfg,
        embed_dim=256,       
        num_encoder_layers=4,
        num_decoder_layers=4,
        nhead=4
    ).to(DEVICE)

    try:
        # Usamos map_location por si hacemos la inferencia en CPU para evitar errores
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
        print(f"Modelo cargado correctamente.")
    except Exception as e:
        print(f"Ocurrió un error al caragr los pesos: {e}")
        return
    
    # Diccionario para las métricas acumuladas
    metrics = {
        'onset': [], 
        'onset_offset': [],
        'velocity': []
        }
    
    # Bucle de evaluación
    # Establecemos un límite para la realizacion de pruebas rápidas,
    # si queremos realizar una evaluación completa simplemente lo ponemos a None
    count = 0
    limit = 10

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df) if limit is None else limit):
        if limit and count >= limit: break

        audio_filename = os.path.join(ROOT_DIR, row['audio_filename'])
        midi_filename_gt = os.path.join(ROOT_DIR, row['midi_filename'])

        if not os.path.exists(audio_filename):
            print(f"Audio no encontrado: {audio_filename}")
            continue

        try:
            # Procesamos el audio
            mel = ap.compute_spectogram(audio_filename) # (n_mels, time)
            audio_tensor = torch.tensor(mel).unsqueeze(0).to(DEVICE) ## (1, n_mels, time)

            # Inferencia
            pred_tokens = predict_greedy(model, audio_tensor, mp)

            # Decodificamos a MIDI
            pred_midi_path = os.path.join(OUTPUT_DIR, f"pred_{count}.mid")
            pred_midi_obj = mp.decode_midi(pred_tokens, output_path=pred_midi_path)

            # Calculamos las métricas
            on_f1, on_off_f1, vel_f1 = calculate_metrics(midi_filename_gt, pred_midi_obj)

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