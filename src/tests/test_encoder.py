import torch
from src.models.encoder import AudioEncoder

# Simular un batch de espectrogramas
# Batch=2, Canal=1, Frecuencias=229, Tiempo=1000 frames
dummy_input = torch.randn(2, 1, 229, 1000)

print("--- Probando Encoder de Audio ---")
print(f"Input shape: {dummy_input.shape}")

# Instanciar modelo
encoder = AudioEncoder(n_mels=229, d_model=512, num_layers=2)

# Ejecutar
output = encoder(dummy_input)

print(f"Output shape: {output.shape}")

# Verificación
# Esperamos [Batch, Tiempo, d_model] -> [2, 1000, 512]
if output.shape == (2, 1000, 512):
    print("✅ El Encoder funciona correctamente. Las dimensiones cuadran.")
else:
    print("❌ Error en las dimensiones de salida.")