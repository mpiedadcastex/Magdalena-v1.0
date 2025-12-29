import torch
import torch.nn as nn
from types import SimpleNamespace # Para simular el objeto de configuración

# Importa tu clase (asegúrate de que la ruta sea correcta)
from models.encoder import AudioSparsifinerEncoder

# --- 1. CREAR UNA CONFIGURACIÓN FALSA (MOCK CONFIG) ---
# El código original del Sparsifiner espera acceder a variables como cfg.SPAR.BASIS_THRESHOLD
# Creamos esta estructura de objetos para que no falle.
def get_dummy_config():
    cfg = SimpleNamespace()
    
    # Configuración de Sparsity (SPAR)
    cfg.SPAR = SimpleNamespace()
    cfg.SPAR.BASIS_THRESHOLD = 0.0
    
    cfg.SPAR.BASIS_COEF = SimpleNamespace()
    cfg.SPAR.BASIS_COEF.THRESHOLD = 0.0
    cfg.SPAR.BASIS_COEF.USE_TOPK = False # Usar top-k o threshold
    cfg.SPAR.BASIS_COEF.TOPK = 32        # Valor dummy
    
    cfg.SPAR.ATTN_SCORE = SimpleNamespace()
    cfg.SPAR.ATTN_SCORE.USE_TOPK = True  # Forzamos usar Top-K para la máscara
    cfg.SPAR.ATTN_SCORE.THRESHOLD = 0.0
    
    cfg.SPAR.PRUNE_ATTN_MATRIX_ROW = False
    
    # Flags de salida para métricas (los ponemos en False para ahorrar cómputo)
    cfg.SPAR.OUT_BASIS_SPARSITY = False
    cfg.SPAR.OUT_BASIS_COEF_SPARSITY = False
    cfg.SPAR.OUT_ATTN_MASK_SPARSITY = False

    # Configuración de Loss (Pérdida)
    cfg.LOSS = SimpleNamespace()
    cfg.LOSS.USE_ATTN_RECON = False # No estamos reconstruyendo atención
    cfg.LOSS.USE_L1 = False
    
    return cfg

def test_encoder():
    print("--- INICIANDO TEST DEL AUDIO ENCODER ---")
    
    # 1. Definir parámetros de prueba
    BATCH_SIZE = 2
    MEL_BINS = 229
    TIME_FRAMES = 1000  # Unos 20 segundos de audio dummy
    EMBED_DIM = 256     # Más pequeño que 512 para la prueba rápida
    
    # 2. Instanciar el Modelo
    print("1. Instanciando el modelo...")
    cfg = get_dummy_config()
    
    try:
        model = AudioSparsifinerEncoder(
            mel_bins=MEL_BINS,
            max_seq_len=2000,   # Margen suficiente
            embed_dim=EMBED_DIM,
            depth=2,            # Solo 2 capas para probar rápido
            num_heads=4,
            reduce_n_factor=4,  # Factor de reducción pequeño para este test
            cfg=cfg             # PASAMOS LA CONFIG DUMMY
        )
        print("   -> Modelo instanciado correctamente.")
    except Exception as e:
        print(f"   -> ERROR instanciando el modelo: {e}")
        return

    # 3. Crear datos falsos (Tensores aleatorios)
    # Forma que sale de tu AudioProcessor: (Batch, 229, Time)
    dummy_input = torch.randn(BATCH_SIZE, MEL_BINS, TIME_FRAMES)
    print(f"2. Datos de entrada generados: {dummy_input.shape}")

    # 4. Forward Pass (Pasada hacia adelante)
    print("3. Ejecutando forward pass...")
    try:
        output = model(dummy_input)
        print("   -> Forward pass completado.")
    except Exception as e:
        print(f"   -> ERROR durante el forward pass: {e}")
        import traceback
        traceback.print_exc() # Esto te dirá la línea exacta del error
        return

    # 5. Verificaciones Finales
    print("4. Verificando dimensiones de salida...")
    
    # Esperamos: (Batch, Time, Embed_Dim) -> (2, 1000, 256)
    expected_shape = (BATCH_SIZE, TIME_FRAMES, EMBED_DIM)
    
    if output.shape == expected_shape:
        print(f"✅ ÉXITO: La salida tiene la forma correcta: {output.shape}")
        print("   El Encoder ha transformado (Batch, Mel, Time) -> (Batch, Time, Dim)")
    else:
        print(f"❌ FALLO: Se esperaba {expected_shape}, pero se obtuvo {output.shape}")

if __name__ == "__main__":
    test_encoder()