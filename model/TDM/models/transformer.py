import math
import torch
import torch.nn as nn
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer import LinearAttentionTransformer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearAttentionTransformerEmbedding(nn.Module):
    def __init__(self, input_dim, 
                 output_dim, 
                 dim, 
                 depth, 
                 n_blocks, 
                 max_seq_len, 
                 num_timesteps, 
                 heads = 8, 
                 dim_head = None, 
                 causal = False, 
                 one_kv_head = False, 
                 reversible = False, 
                 ff_chunks = 1, 
                 ff_glu = False, 
                 ff_dropout = 0., 
                 attn_layer_dropout = 0., 
                 attn_dropout = 0., 
                 blindspot_size = 1, 
                 n_local_attn_heads = 0, 
                 local_attn_window_size = 128, 
                 return_embeddings = False, 
                 receives_context = True, 
                 pkm_layers = tuple(), 
                 pkm_num_keys = 128, 
                 attend_axially = False, 
                 linformer_settings = None, 
                 context_linformer_settings = None, 
                 context_mask_flag=False):
        assert (max_seq_len % local_attn_window_size) == 0, 'max sequence length must be divisible by the window size, to calculate number of kmeans cluster'
        super().__init__()
        self.max_seq_len = max_seq_len
        self.depth = depth
        emb_dim = dim
        self.emb_dim = emb_dim
        self.n_blocks = n_blocks

        self.first = nn.Embedding(input_dim, emb_dim)

        self.context_mask_flag = context_mask_flag
        if self.context_mask_flag:
            self.post_embed = nn.Linear(emb_dim*2, emb_dim)

        self.time_pos_emb = SinusoidalPosEmb(emb_dim, num_timesteps)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.Softplus(),
            nn.Linear(emb_dim * 4, emb_dim * n_blocks * depth)
        )

        self.axial_pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape=(max_seq_len // local_attn_window_size, local_attn_window_size))

        self.transformer_blocks = torch.nn.ModuleList()
        for i in range(n_blocks):
            self.transformer_blocks.append(torch.nn.ModuleList())
            for j in range(depth):
                self.transformer_blocks[-1].append(
                    LinearAttentionTransformer(
                        dim, 1, max_seq_len, heads = heads, dim_head = dim_head,
                        causal = causal, 
                        ff_chunks = ff_chunks, ff_glu = ff_glu,
                        ff_dropout = ff_dropout,
                        attn_layer_dropout = attn_layer_dropout,
                        attn_dropout = attn_dropout, reversible = reversible,
                        blindspot_size = blindspot_size,
                        n_local_attn_heads = n_local_attn_heads,
                        local_attn_window_size = local_attn_window_size,
                        receives_context = receives_context,
                        pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys,
                        attend_axially = attend_axially,
                        linformer_settings = linformer_settings,
                        context_linformer_settings = context_linformer_settings))

        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(emb_dim, output_dim) if not return_embeddings else nn.Identity()

    def forward(self, x, t, context_image, context_mask=None):
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        time_embed = t.view(x.size(0), 1, self.emb_dim, self.n_blocks, self.depth)

        x = self.first(x)
        if self.context_mask_flag:
            context_mask = self.first(context_mask)
            x = torch.cat((x, context_mask), dim=-1)
            x = self.post_embed(x)

        x_embed_axial = x + self.axial_pos_emb(x).type(x.type())
        h = torch.zeros_like(x_embed_axial)

        for i, block in enumerate(self.transformer_blocks):
            h = h + x_embed_axial
            for j, transformer in enumerate(block):
                h = transformer(h + time_embed[..., i, j], context=context_image)

        h = self.norm(h)
        return self.out(h)
