import pretty_midi
import numpy as np

class MidiProcessor:
    def __init__(self):
        """
        Clase de creacion de vocabulario y tokenizacion de MIDI_
        
        """
        #  ------------ CREACIÓN DEL VOCABULARIO ------------  
        # 1. Definición de constantes
        # PITCH
        self.MIN_PITCH = 21   # La tecla más baja del piano (A0) en notación MIDI 
        self.MAX_PITCH = 108  # La tecla más alta del piano (C8) en notación MIDI
        self.NUM_PITCHES = 88  # 88 teclas en total

        # TIME
        self.TIME_STEP = 0.01  # Paso de tiempo en segundos (10ms)
        self.TIME_BINS = 100   # Hasta 1 segundo de espera (100 * 10ms)
        self.TIME_SHIFTS = int(self.TIME_BINS/self.TIME_STEP)  # Número de pasos de tiempo

        # VELOCITY
        self.VELOCITY_BINS = 32  # Niveles de velocidad 

        # 2. Rangos de índices de tokens

        # Padding -> 0
        self.token_pad = 0

        # Note-On ->  Lo establecemos desde 1 y no desde 0 por motivos de intuitividad
        # Rango: 1 a 88
        self.idx_note_on = 1

        # Note-Off -> 
        # Rango: 89 a 176
        self.idx_note_off = self.idx_note_on + self.NUM_PITCHES # 1 + 88 = 89
        
        # Time-Shift -> 
        # Rango: 177 a 276
        self.idx_time = self.idx_note_off + self.NUM_PITCHES  # 89 + 88 = 177

        # Velocity -> 
        # Rango: 277 a 308 
        self.idx_vel = self.idx_time + self.TIME_BINS  # 177 + 100 = 277
        
        # Comienzo (Start of Sequence) y Fin (End of Sequence) de la secuencia 
        self.token_sos = self.idx_vel + self.VELOCITY_BINS    # 277 + 32 = 309
        self.token_eos = self.token_sos + 1                         # 309 + 1 = 310

        # Tamaño total del vocabulario
        self.vocab_size = self.token_eos +1   # 310 + 1 = 311
        
        self.token_to_id = {
            '<pad>': self.token_pad,
            '<sos>': self.token_sos,
            '<eos>': self.token_eos
        }

    def encode_midi(self, midi):
        """
        Convierte un archivo MIDI en una secuencia de tokens basada en los eventos creados.
        
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi)
        except Exception as e:
            print(f"Error al cargar MIDI: {e}")
            return np.array([self.token_sos, self.token_eos], dtype=np.int32)
            
        # 1. Extraemos todas las notas
        notes = []
        for instrument in midi_data.instruments: # En nuestro caso es un solo instrumento
            for note in instrument.notes:
                notes.append(note)

        # 2. Ordenamos notas por tiempo de inicio para su procesamiento cronológico
        notes.sort(key=lambda x: x.start)

##############################################################################################################

        # 3. Etapa de GENERACIÓN DE EVENTOS a partir de las notas MIDI
        events = []

        # Desglosamos cada nota en dos eventos separados:
        # 1. Momento en que comienza (Note On)
        # 2. Momento en que termina (Note Off)

        for note in notes:
            # Primero filtramos las notas fuera del rango del piano para no tenerlas en cuenta
            if not (self.MIN_PITCH <= note.pitch <= self.MAX_PITCH):
                continue

            # Note On
            events.append({
                'type': 'on',
                'pitch': note.pitch,
                'time': note.start,
                'velocity': note.velocity
                })
            
            # Note Off
            events.append({
                'type': 'off',
                'pitch': note.pitch,
                'time': note.end,
                'velocity': 0  # El evento Note Off no tiene velocity
                })
            
        # Ordenación de los eventos por tipo y tiempo
        # 1º Ordenamos todos los eventos por tiempo
        # 2º Si hay empate, Note Off (0) antes que Note On (1)
        events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'off' else 1))

###############################################################################################################

        # 4. Etapa de CONVERSIÓN de eventos A TOKENS
        tokens = [self.token_sos]

        current_time = 0.0

        for event in events:
            # 1. Calculamos el tiempo transcurrido entre el nuevo evento y la última marca 
            # de tiempo registrada (current_time) 
            time_shift = event['time'] - current_time

            # Si ha pasado el suficiente tiempo (conforme a nuestra constante TIME_STEP) desde el último 
            # evento, añadimos tokens de tiempo
            steps = int(round(time_shift / self.TIME_STEP))

            while steps > 0:
                take_steps = min(steps, self.TIME_BINS)
                token_index = self.idx_time + (take_steps - 1) # -1 porque self.idx_time + 0 ya añade un step de tiempo
                tokens.append(token_index)
                steps -= take_steps

            # Actualizamos el tiempo actual
            current_time = event['time']

            # Gestion de la velocity -> SOLO para Note On
            if event['type'] == 'on':
                # Mapear la velocidad a uno de los VELOCITY_BINS
                # Nota: Dividimos por 128 ya que es el número de bins que establece MIDI 
                # para codificar la velocidad 
                vel_index = int((event['velocity'] / 128) * (self.VELOCITY_BINS))
                vel_index = min(vel_index, self.VELOCITY_BINS - 1)
                tokens.append(self.idx_vel + vel_index)

            # Calculamos el índice de pitch, en un rango de 1 a 88 
            pitch_index = event['pitch'] - self.MIN_PITCH 
            
            # Usamos el índice para, dependiendo del tipo de evento,
            # calcular el token correspondiente
            if event['type'] == 'on':
                # Para evento NOTE_ON 
                tokens.append(self.idx_note_on + pitch_index)
            else:
                tokens.append(self.idx_note_off + pitch_index)

        # Finalizamos la secuencia con el token <EOS>
        tokens.append(self.token_eos)
   
        return np.array(tokens, dtype=np.int32)
    


    def decode_midi(self, tokens, output_path=None):
        """
        Convierte los tokens a MIDI (ignorando PAD, SOS y EOS)
        """
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)

        current_time = 0.0
        # CORRECCIÓN: Velocidad por defecto a 64 (intensidad media) por si el modelo
        # "alucina" un Note On antes de definir la velocidad.
        current_velocity = 64

        # Diccionario para el rastreo de las notas activas
        # Clave: Pitch -> Valor:(start_time, velocity)
        active_notes = {}

        for token in tokens:
            token = int(token)

            # Tokens especiales: PAD, SOS Y EOS
            if token == self.token_sos or token == self.token_eos or token == self.token_pad:
                continue
            
            # INICIO DE NOTA (NOTE ON)
            elif self.idx_note_on <= token < self.idx_note_off:
                pitch_index = token - self.idx_note_on
                pitch = pitch_index + self.MIN_PITCH

                # IMPORTANTE: si ya estaba la misma nota sonando, se cierra forzosamente 
                # antes de comenzar la nueva (no se puede tocar la misma tecla 2 veces al mismo tiempo)
                if pitch in active_notes:
                    start, vel = active_notes[pitch]

                    # Creamos la nota MIDI y la añadimos a la secuencia
                    note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=current_time)
                    piano.notes.append(note)

                # Añadimos la nueva nota al diccionario
                active_notes[pitch] = (current_time, current_velocity)

            # FINAL DE NOTA (NOTE OFF)
            elif self.idx_note_off <= token < self.idx_time:
                pitch_index = token - self.idx_note_off
                pitch = self.MIN_PITCH + pitch_index

                # Comprobamos que la nota estuviese sonando 
                if pitch in active_notes:
                    start, vel = active_notes[pitch]

                    # Creamos la nota MIDI y la añadimos a la secuencia
                    note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=current_time)
                    note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=current_time)
                    piano.notes.append(note)

                    # La quitamos del diccionario de notas activas
                    del active_notes[pitch]

            # AVANCE DE TIEMPO (TIME SHIFT)
            elif self.idx_time <= token < self.idx_vel:
                time_index = token - self.idx_time

                # Calculamos los pasos teniendo en cuenta que time_index = 0 equivale a 1 step
                steps = time_index + 1
                time_shift = steps * self.TIME_STEP
                current_time += time_shift

            # DINÁMICA (VELOCITY)
            elif self.idx_vel <= token < (self.idx_vel + self.VELOCITY_BINS):
                vel_index = token - self.idx_vel
                # Deshacemos la normalizacion para volver al rango original de 0-127
                # Formula inversa: (index / bins) * 128
                value = int((vel_index / self.VELOCITY_BINS) * 128)
                # Clamp por seguridad para matenerlo en rango MIDI válido
                current_velocity = max(1, min(127, value))

        # -- LIMPIEZA FINAL --
        # Si al termianr la secuencia quedaron notas abiertas las cerramos
        for pitch, (start, vel) in active_notes.items():
            note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=current_time)
            piano.notes.append(note)

        # Añadimos el instrumento al objeto MIDI
        midi.instruments.append(piano)

        # Guardar archivo si se especifica ruta
        if output_path:
            try:
                midi.write(output_path)
                print(f"MIDI guardado exitosamente en : {output_path}")
            except Exception as e:
                print(f"Error al guardar MIDI: {e}")

        return midi