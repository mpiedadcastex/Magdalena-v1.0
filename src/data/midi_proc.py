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
        # Note-On -> 1 a 88     Lo establecemos desde 1 y no desde 0 por motivos de intuitividad
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
        #print(f"Procesador MIDI inicializado con vocabulario de tamaño: {self.vocab_size}")

    def process_midi(self, midi):
        """
        Convierte un archivo MIDI en una secuencia de tokens basada en los eventos creados.
        
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi)
        except Exception as e:
            print(f"Error al cargar MIDI: {e}")
            return []
        
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
                'velocity': 0  # El evento Note Off no tiene velocity
                })
            
        # Ordenación de los eventos por tipo y tiempo
        # 1º Ordenamos todos los eventos por tiempo
        # 2º Si hay empate, Note Off (0) antes que Note On (1)
        events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'off' else 1))

###############################################################################################################

        # 4. Estapa de CONVERSIÓN de eventos A TOKENS
        tokens = []
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
                # Nota: Dividimos por 128 ya que es el numero de bins que establece MIDI 
                # para codificar la velocidad (le asigna 7 bytes = 2^7 = 128 bins)
                vel_index = int((event['velocity'] / 128) * (self.VELOCITY_BINS))
                tokens.append(self.idx_vel + vel_index)

            # Calculamos el índice de pitch, en un rango de 0 a 87, 
            pitch_index = event['pitch'] - self.MIN_PITCH  
            
            # Usamos el índice para, dependiendo del tipo de evento,
            # calcular el token correspondiente
            if event['type'] == 'on':
                # Para evento NOTE_ON 
                tokens.append(self.idx_note_on + pitch_index)
            else:
                tokens.append(self.idx_note_off + pitch_index)
    
        return np.array(tokens, dtype=np.int32)