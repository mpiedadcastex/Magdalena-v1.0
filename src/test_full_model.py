import torch
from models.transformer import PianoTranscriptionModel

print("--- Probando Modelo Completo (End-to-End) ---")

# 1. Configuración (Hiperparámetros de prueba)
config = {
    'n_mels': 229,
    'd_model': 256,          # Reducido para que el test sea rápido
    'nhead': 4,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'dim_feedforward': 512,
    'vocab_size': 350,       # Tamaño aproximado de nuestro vocabulario MIDI
    'dropout': 0.1
}

# 2. Instanciar el Modelo
model = PianoTranscriptionModel(config)
print("✅ Modelo instanciado correctamente.")

# 3. Simular Datos (Batch Size = 2)
# Audio: [Batch, 1, Frecuencias, Tiempo]
audio_input = torch.randn(2, 1, 229, 500) 

# Tokens Objetivo (Entrada del Decoder): [Batch, Secuencia]
# Simulamos una secuencia de 100 tokens
tgt_input = torch.randint(0, 350, (2, 100))

print(f"Input Audio: {audio_input.shape}")
print(f"Input Tokens: {tgt_input.shape}")

# 4. Forward Pass (Entrenamiento)
logits = model(audio_input, tgt_input)

print(f"Output Logits: {logits.shape}")

# 5. Validaciones
expected_shape = (2, 100, 350) # [Batch, Seq_Len, Vocab_Size]

if logits.shape == expected_shape:
    print("✅ INTEGRACIÓN EXITOSA: El audio fluye hasta la predicción de tokens.")
else:
    print(f"❌ ERROR: Dimensiones incorrectas. Esperado {expected_shape}, recibido {logits.shape}")

# 6. Prueba rápida de Inferencia (Greedy)
print("\n--- Prueba de Inferencia (Greedy) ---")
# Generar 5 tokens a partir del audio
generated = model.generate(audio_input[0:1], start_token=1, max_len=5) 
print(f"Tokens generados: {generated}")