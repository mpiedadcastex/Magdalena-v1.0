import torch
import torch.nn as nn
from .sparsifiner import Block, trunc_normal_
    
class AudioSparsifinerEncoder(nn.Module):
    def __init__(
            self,
            mel_bins=229,
            max_seq_len=70000,
            embed_dim=512,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
            attn_keep_rate=0.25,
            reduce_n_factor=16,
            drop_rate=0.1,
            norm_layer=nn.LayerNorm,
            cfg=None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.cfg = cfg

        # 1º Proyeccion Lineal (Es donde podríamos poner la CNN)
        self.input_proj = nn.Linear(mel_bins, embed_dim)

        # 2º Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3º Bloques Sparsifiner
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_tokens=max_seq_len,
                num_heads=num_heads,
                attn_keep_rate=attn_keep_rate,
                token_keep_rate=1.0,
                token_pruning_this_layer=False,
                reduce_n_factor=reduce_n_factor,
                reduce_c_factor=2,
                share_inout_proj=False,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                norm_layer=norm_layer,
                cfg=cfg
            )
            for i in range(depth)   # Creamos una lista de 6 bloques 
        ])

        self.norm = norm_layer(embed_dim)

        # Inicialización de pesos
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        # Forma inicial de x : (Batch, 229, Time)

        # Primero permutamos las posiciones de 229 (n_mels) y Time
        # x: (Batch, 229, Time) -> (Batch, Time, 229)
        x = x.permute(0, 2, 1)

        # Ahora ya podemos proyectar:
        # x: (Batch, Time, 229) -> (Batch, Time, 512)
        x = self.input_proj(x)

        # Añadimos las posiciones
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.pos_drop(x)

        # Pasamos por los bloques que hemos creado de sparsifiner
        token_mask = None

        for block in self.blocks:
            out_dict = block(x, token_mask)
            x = out_dict['x']
            token_mask = out_dict['token_mask']

        # Normalizamos al final
        x = self.norm(x)

        return x