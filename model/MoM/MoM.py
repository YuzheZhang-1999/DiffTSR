import torch
import torch.nn as nn

from model.IDM.utils.util import instantiate_from_config


class MoM_model(nn.Module):
    def __init__(self,
                 MoM_Unet=None,
                 MoM_Transformer=None,
                 scale_factor=0.18215,
                 num_classes=6736,
                 context_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.first_embeder = nn.Embedding(num_classes, context_dim)
        self.Unet = instantiate_from_config(MoM_Unet)
        self.Transformer = instantiate_from_config(MoM_Transformer)
        self.scale_factor = scale_factor

    def forward(self, c_t, z_lr=None, z_t=None, t=None):
        z_input = torch.cat((z_lr, z_t), dim=1)
        z_input *= self.scale_factor
        c_t_embed = self.first_embeder(c_t)
        I_cond = self.Unet(z_input, timesteps=t, context=c_t_embed)
        C_cond = self.Transformer(c_t)

        return I_cond, C_cond
        

