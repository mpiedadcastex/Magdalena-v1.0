from types import SimpleNamespace

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