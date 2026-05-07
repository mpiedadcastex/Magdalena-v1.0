import os
from datetime import datetime
from types import SimpleNamespace

def log_training_loss(epoch, avg_loss, log_interval=1, base_path="/content/drive/MyDrive/TFG_Project/MPCS/checkpoints/v1_baseline/"):
    """
    Registra el Loss promedio en un archivo de texto con formato elegante.
    
    Args:
        epoch (int): El número de epoch actual (base 0).
        avg_loss (float): El valor del loss promedio.
        log_interval (int): Cada cuántas epochs se debe guardar (ej: 1 para todas, 5 para cada 5).
        base_path (str): Ruta donde guardar el log.
    """
    
    # 1. Verificar si toca guardar en este epoch
    # Sumamos 1 al epoch porque suelen empezar en 0, pero queremos contar 1, 2, 3...
    if (epoch + 1) % log_interval != 0:
        return

    # 2. Asegurar que el directorio existe
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, "training_loss_log.txt")
    
    # 3. Preparar los datos
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch_num = epoch + 1
    
    # Formato visual:
    # - <20 significa alineado a la izquierda con 20 espacios
    # - ^10 significa centrado en 10 espacios
    # - .6f significa 6 decimales de precisión
    log_entry = f"| {timestamp:<20} | Epoch: {epoch_num:^8} | Loss: {avg_loss:<12.6f} |\n"
    separator = "+" + "-"*22 + "+" + "-"*16 + "+" + "-"*20 + "+\n"

    # 4. Escribir en el archivo (Modo 'a' = append)
    new_file = not os.path.exists(file_path)
    
    with open(file_path, "a", encoding="utf-8") as f:
        # Si es archivo nuevo, escribimos la cabecera
        if new_file:
            header_top = "+" + "="*22 + "+" + "="*16 + "+" + "="*20 + "+\n"
            header_txt = f"| {'TIMESTAMP':^20} | {'EPOCH':^16} | {'AVG LOSS':^18} |\n"
            f.write(header_top)
            f.write(header_txt)
            f.write(header_top)
        
        # Escribimos el registro
        f.write(log_entry)
        # Escribimos una línea separadora para dar ese "espacio armonioso" y claridad
        f.write(separator)

    print(f"   -> Log guardado en: {file_path}")

def get_model_config():
    """
    Configuración que el Sparsifiner necesita para funcionar.
    """
    cfg = SimpleNamespace()
    
    # Sparsity Config
    cfg.SPAR = SimpleNamespace()
    cfg.SPAR.BASIS_THRESHOLD = 0.0
    cfg.SPAR.BASIS_COEF = SimpleNamespace()
    cfg.SPAR.BASIS_COEF.THRESHOLD = 0.0
    cfg.SPAR.BASIS_COEF.USE_TOPK = False 
    cfg.SPAR.BASIS_COEF.TOPK = 32        
    cfg.SPAR.ATTN_SCORE = SimpleNamespace()
    cfg.SPAR.ATTN_SCORE.USE_TOPK = True  
    cfg.SPAR.ATTN_SCORE.THRESHOLD = 0.0
    cfg.SPAR.PRUNE_ATTN_MATRIX_ROW = False
    cfg.SPAR.OUT_BASIS_SPARSITY = False
    cfg.SPAR.OUT_BASIS_COEF_SPARSITY = False
    cfg.SPAR.OUT_ATTN_MASK_SPARSITY = False

    # Loss Config
    cfg.LOSS = SimpleNamespace()
    cfg.LOSS.USE_ATTN_RECON = False 
    cfg.LOSS.USE_L1 = False
    
    return cfg
