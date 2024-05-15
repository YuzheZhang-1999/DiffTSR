import torch
import torch.nn as nn

from model.TDM.models.transformer import LinearAttentionTransformerEmbedding 


class Rezero(torch.nn.Module):
    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x


class DynamicsTransformer(nn.Module):
        def __init__(self, 
                     num_classes, 
                     transformer_dim=768,
                     transformer_heads=16,
                     transformer_depth=12,
                     transformer_blocks=1,
                     max_seq_len=24,
                     diffusion_steps=1000,
                     attn_layer_dropout=0.0,
                     transformer_local_heads=8,
                     transformer_local_size=4,
                     transformer_reversible=False,
                     receives_context=True,
                     context_mask_flag=False,
                     ):
            super(DynamicsTransformer, self).__init__()
            self.transformer = LinearAttentionTransformerEmbedding(
                input_dim=num_classes,
                output_dim=num_classes,
                dim=transformer_dim,
                heads=transformer_heads,
                depth=transformer_depth,
                n_blocks=transformer_blocks,
                max_seq_len=max_seq_len,
                num_timesteps=diffusion_steps,
                causal=False,  # auto-regressive or not
                ff_dropout=0,  # dropout for feedforward
                attn_layer_dropout=attn_layer_dropout,
                # dropout right after self-attention layer
                attn_dropout=0,  # dropout post-attention
                n_local_attn_heads=transformer_local_heads,
                # number of local attention heads for (qk)v attention.
                # this can be a tuple specifying the exact number of local
                # attention heads at that depth
                local_attn_window_size=transformer_local_size,
                # receptive field of the local attention
                reversible=transformer_reversible,
                # use reversible nets, from Reformer paper
                receives_context = receives_context,
                context_mask_flag = context_mask_flag
            )

            self.rezero = Rezero()

        def forward(self, x, t, context_image, context_mask=None):
            x = self.transformer(x, t, context_image=context_image, context_mask=context_mask)
            x = x.permute(0, 2, 1)
            x = self.rezero(x)
            return x
