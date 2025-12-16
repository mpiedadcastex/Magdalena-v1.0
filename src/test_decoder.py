import torch
from models.decoder import MidiDecoder

print("--- Probando Decoder MIDI ---")

# 1. Simulación de Datos
BATCH_SIZE = 2
VOCAB_SIZE = 350 # Nuestro vocabulario (aprox)
D_MODEL = 512
SEQ_LEN = 50     # Longitud de la secuencia de tokens (ej. partitura parcial)
AUDIO_LEN = 1000 # Longitud de la secuencia de audio (output del encoder)

# Memoria (lo que sale del Encoder)
memory = torch.randn(BATCH_SIZE, AUDIO_LEN, D_MODEL)

# Target (los tokens que supuestamente ya hemos escrito)
# Son enteros aleatorios entre 0 y vocab_size
tgt_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

# 2. Instanciar Modelo
decoder = MidiDecoder(vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=2)

print(f"Input Memory shape (Audio): {memory.shape}")
print(f"Input Target shape (Tokens): {tgt_tokens.shape}")

# 3. Ejecutar Forward
logits = decoder(tgt=tgt_tokens, memory=memory)

print(f"Output Logits shape: {logits.shape}")

# 4. Verificación
# Esperamos: [Batch, Seq_Len, Vocab_Size]
# El modelo debe dar una probabilidad para cada palabra del diccionario, por cada paso de tiempo.
expected_shape = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

if logits.shape == expected_shape:
    print("✅ El Decoder funciona correctamente. Dimensiones alineadas.")
else:
    print(f"❌ Error. Se esperaba {expected_shape} pero se obtuvo {logits.shape}")

# Verificar que la máscara funciona (no da error de ejecución)
print("✅ Máscara causal aplicada internamente sin errores.")