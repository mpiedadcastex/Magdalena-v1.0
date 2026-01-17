from data.midi_proc import MidiProcessor
import numpy as np

# Inicializar
proc = MidiProcessor()

# Crear un archivo MIDI falso para probar (un Do mayor simple)
import pretty_midi
piano_midi = pretty_midi.PrettyMIDI()
piano_inst = pretty_midi.Instrument(program=0)

# Nota 1: Do (60) empieza en 0.0s, dura 0.5s, velocidad 100
note1 = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5)
# Nota 2: Mi (64) empieza en 0.5s (justo al acabar la anterior), dura 0.5s
note2 = pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=1.0)

piano_inst.notes.append(note1)
piano_inst.notes.append(note2)
piano_midi.instruments.append(piano_inst)
piano_midi.write('test_scale.mid')

# PROCESAR
print("\n--- Procesando MIDI ---")
tokens = proc.encode_midi('test_scale.mid')

print(f"Tokens generados: {tokens}")
print("\n--- Decodificación Manual (Lectura) ---")
# Vamos a traducir los números a mano para ver si tiene sentido
for t in tokens:
    if proc.idx_vel <= t < proc.idx_vel + proc.VELOCITY_BINS:
        val = t - proc.idx_vel
        print(f"TOKEN {t}: Velocidad {val}/32")
    elif proc.idx_note_on <= t < proc.idx_note_on + proc.NUM_PITCHES:
        note = t - proc.idx_note_on + proc.MIN_PITCH
        print(f"TOKEN {t}: NOTE ON (Tecla {note})")
    elif proc.idx_time <= t < proc.idx_time + proc.TIME_BINS:
        ms = (t - proc.idx_time + 1) * 10
        print(f"TOKEN {t}: ESPERAR {ms}ms")
    elif proc.idx_note_off <= t < proc.idx_note_off + proc.NUM_PITCHES:
        note = t - proc.idx_note_off + proc.MIN_PITCH
        print(f"TOKEN {t}: NOTE OFF (Soltar {note})")