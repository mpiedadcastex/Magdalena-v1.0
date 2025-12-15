import pretty_midi
import numpy as np

class MidiProcessor:
    def __init__(self):
        """
        Traductor de MIDI a Tokens -> Representacion basada en eventos_
        
        """
        # --- DEFINICIÓN DEL VOCABULARIO ---
        self.MIN_PITCH = 21   # La tecla más baja del piano (A0) en notación MIDI 
        self.MAX_PITCH = 108  # La tecla más alta del piano (C8) en notación MIDI
        self.NUM_PITCHES = 88  # 88 teclas en total

        self.TIME_STEP = 0.01  # Paso de tiempo en segundos (10ms)
        self.TIME_BINS = 100   # Hasta 1 segundo de espera (100 * 10ms)
        self.TIME_SHIFTS = int(self.TIME_BINS/self.TIME_STEP)  # Número de pasos de tiempo

        self.VELOCITY_BINS = 32  # Niveles de velocidad 

        # --- RANGOS DE ÍNDICES DE TOKENS ---
        # Note-On -> 1 a 88
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
        
        # Tamaño total del vocabulario
        self.vocab_size = self.idx_vel + self.VELOCITY_BINS
        print(f"Procesador MIDI inicializado con vocabulario de tamaño: {self.vocab_size}")

    def process_midi(self, midi):
        """
        Convierte un archivo MIDI en una secuencia de tokens basada en eventos.
        
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi)
        except Exception as e:
            print(f"Error al cargar MIDI: {e}")
            return []
        
        # Extraer todas las notas
        notes = []
        for instrument in midi_data.instruments: # En nuestro caso es un solo instrumento
            for note in instrument.notes:
                notes.append(note)

        # Ordenar notas por tiempo de inicio -> Procesamos la música cronológicamente
        notes.sort(key=lambda x: x.start)

        # --- GENERACIÓN DE EVENTOS ---
        events = []

        # Desglosamos cada nota en dos eventos separados:
        # 1. Momento en que comienza (Note On)
        # 2. Momento en que termina (Note Off)

        for note in notes:
            # Primero filtramos las notas fuera del rango del piano
            if not (self.MIN_PITCH <= note.pitch <= self.MAX_PITCH):
                continue

            # Evento Note On
            events.append({
                'type': 'on',
                'pitch': note.pitch,
                'time': note.start,
                'velocity': note.velocity
                })
            
            # Evento Note Off
            events.append({
                'type': 'off',
                'pitch': note.pitch,
                'time': note.end,
                'velocity': 0  # Velocity no importa para Note Off
                })
            
        # Ordenar eventos por tiempo
        # 1º Ordenamos todos los eventos por tiempo
        # 2º Si hay empate, Note Off (0) antes que Note On (1)
        events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'off' else 1))

        # --- CONVERSIÓN A TOKENS ---
        tokens = []
        current_time = 0.0

        for event in events:
            # 1. Gestionar el paso del tiempo (TIME SHIFT)
            time_delta = event['time'] - current_time

            # Si ha pasado tiempo desde el último evento, añadimos tokens de tiempo
            steps = int(round(time_delta / self.TIME_STEP))

            while steps > 0:
                take_steps = min(steps, self.TIME_BINS)
                token_index = self.idx_time + (take_steps - 1) # -1 porque es base 0
                tokens.append(token_index)
                steps -= take_steps

            # Actualizamos el tiempo actual
            current_time = event['time']

            # 2. Gestion de la Velocidad (SOLO para Note On)
            if event['type'] == 'on':
                # Mapear la velocidad a un bin
                vel_index = int((event['velocity'] / 127) * (self.VELOCITY_BINS - 1))
                tokens.append(self.idx_vel + vel_index)

            # 3. Añadir el evento Note On o Note Off
            pitch_index = event['pitch'] - self.MIN_PITCH  # Normalizamos a 0-87
            if event['type'] == 'on':
                tokens.append(self.idx_note_on + pitch_index)
            else:
                tokens.append(self.idx_note_off + pitch_index)
    
        return np.array(tokens, dtype=np.int32)