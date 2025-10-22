import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
import torchvision.transforms as transforms

from model.IDM.utils.util import instantiate_from_config, default
from model.IDM.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps
from model.IDM.modules.diffusionmodules.util import make_beta_schedule, noise_like, extract_into_tensor

from model.TDM.models.diffusion_multinomial import extract, log_add_exp, cosine_beta_schedule
from model.TDM.models.diffusion_multinomial import log_1_min_a, index_to_log_onehot, log_onehot_to_index


class DiffTSR_pipline(object):
    def __init__(self,
                 max_length = 24,
                 num_classes = 6736,
                 transformer_dim = 768,
                 scale_factor = 0.18215,
                 IDM_Unet_config = None,
                 TDM_Decoder_config = None,
                 MoM_module_config = None,
                 VAE_model_config = None,
                 Text_Prediction_config = None):
        super(DiffTSR_pipline, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.max_length = max_length
        self.num_classes = num_classes
        self.transformer_dim = transformer_dim
        self.scale_factor = scale_factor

        self.IDM_Unet = instantiate_from_config(IDM_Unet_config).to(self.device).eval()
        self.TDM_Decoder = instantiate_from_config(TDM_Decoder_config).to(self.device).eval()
        self.MoM_module = instantiate_from_config(MoM_module_config).to(self.device).eval()
        self.VAE_model = instantiate_from_config(VAE_model_config).to(self.device).eval()
        self.Text_Prediction = instantiate_from_config(Text_Prediction_config).to(self.device).eval()

        self.IDM_ddpm_schedule_init(timesteps=1000, schedule="linear", 
                                    linear_start=0.0015, linear_end=0.0205, cosine_s=8e-3)
                                    
        self.IDM_ddim_schedule_init(ddim_num_steps=200, ddim_discretize="uniform",
                                    ddim_eta=0.2, verbose=False)

        self.TDM_ddpm_schedule_init(timesteps=1000)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.IDM_Unet.load_state_dict(state_dict['IDM_Unet'], strict=True)
        self.TDM_Decoder.load_state_dict(state_dict['TDM_Decoder'], strict=True)
        self.MoM_module.load_state_dict(state_dict['MoM_module'], strict=True)
        self.VAE_model.load_state_dict(state_dict['VAE_model'], strict=True)
        print(f"DiffTSR Model loaded from {path}")

    def IDM_ddpm_schedule_init(self, timesteps, schedule, linear_start, linear_end, cosine_s):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.IDM_betas = make_beta_schedule(schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        self.IDM_alphas = 1. - self.IDM_betas
        self.IDM_alphas_cumprod = np.cumprod(self.IDM_alphas, axis=0)
        self.IDM_alphas_cumprod_prev = np.append(1., self.IDM_alphas_cumprod[:-1])

        self.IDM_betas = to_torch(self.IDM_betas)
        self.IDM_alphas = to_torch(self.IDM_alphas)
        self.IDM_alphas_cumprod = to_torch(self.IDM_alphas_cumprod)
        self.IDM_alphas_cumprod_prev = to_torch(self.IDM_alphas_cumprod_prev)

        timesteps, = self.IDM_betas.shape
        self.IDM_ddpm_num_timesteps = int(timesteps)

    def IDM_ddim_schedule_init(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.IDM_ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.IDM_ddpm_num_timesteps,verbose=verbose)

        assert self.IDM_alphas_cumprod.shape[0] == self.IDM_ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('IDM_betas', to_torch(self.IDM_betas))
        self.register_buffer('IDM_alphas_cumprod', to_torch(self.IDM_alphas_cumprod))
        self.register_buffer('IDM_alphas_cumprod_prev', to_torch(self.IDM_alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('IDM_sqrt_alphas_cumprod', to_torch(np.sqrt(self.IDM_alphas_cumprod.cpu())))
        self.register_buffer('IDM_sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - self.IDM_alphas_cumprod.cpu())))
        self.register_buffer('IDM_log_one_minus_alphas_cumprod', to_torch(np.log(1. - self.IDM_alphas_cumprod.cpu())))
        self.register_buffer('IDM_sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / self.IDM_alphas_cumprod.cpu())))
        self.register_buffer('IDM_sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / self.IDM_alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        IDM_ddim_sigmas, IDM_ddim_alphas, IDM_ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.IDM_alphas_cumprod.cpu(),
                                                                                                ddim_timesteps=self.IDM_ddim_timesteps,
                                                                                                eta=ddim_eta,verbose=verbose)
        self.register_buffer('IDM_ddim_sigmas', IDM_ddim_sigmas)
        self.register_buffer('IDM_ddim_alphas', IDM_ddim_alphas)
        self.register_buffer('IDM_ddim_alphas_prev', IDM_ddim_alphas_prev)
        self.register_buffer('IDM_ddim_sqrt_one_minus_alphas', np.sqrt(1. - IDM_ddim_alphas))
        IDM_sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.IDM_alphas_cumprod_prev) / (1 - self.IDM_alphas_cumprod) * (
                        1 - self.IDM_alphas_cumprod / self.IDM_alphas_cumprod_prev))
        self.register_buffer('IDM_ddim_sigmas_for_original_num_steps', IDM_sigmas_for_original_sampling_steps)

    def TDM_ddpm_schedule_init(self, timesteps):
        TDM_alphas = cosine_beta_schedule(timesteps)
        TDM_alphas = torch.tensor(TDM_alphas.astype('float64'))
        TDM_log_alpha = np.log(TDM_alphas)
        TDM_log_cumprod_alpha = np.cumsum(TDM_log_alpha)
        TDM_log_1_min_alpha = log_1_min_a(TDM_log_alpha)
        TDM_log_1_min_cumprod_alpha = log_1_min_a(TDM_log_cumprod_alpha)

        self.register_buffer('TDM_log_alpha', TDM_log_alpha.float())
        self.register_buffer('TDM_log_1_min_alpha', TDM_log_1_min_alpha.float())
        self.register_buffer('TDM_log_cumprod_alpha', TDM_log_cumprod_alpha.float())
        self.register_buffer('TDM_log_1_min_cumprod_alpha', TDM_log_1_min_cumprod_alpha.float())

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def IDM_p_sample(self, z_t, z_cond, c, t, index, temperature=1.):
        b, *_, device = *z_t.shape, self.device
        e_t = self.IDM_Unet(x=torch.cat([z_t, z_cond], dim=1), timesteps=t, context=c)

        alphas = self.IDM_ddim_alphas
        alphas_prev =  self.IDM_ddim_alphas_prev
        sqrt_one_minus_alphas = self.IDM_ddim_sqrt_one_minus_alphas
        sigmas = self.IDM_ddim_sigmas
        
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        pred_x0 = (z_t - sqrt_one_minus_at * e_t) / a_t.sqrt()

        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(z_t.shape, device, False) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    def IDM_q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.IDM_sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.IDM_sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    @torch.no_grad()
    def TDM_p_sample(self, log_x, t, context):
        model_log_prob = self.TDM_p_pred(log_x=log_x, t=t, context=context)
        out = self.TDM_log_sample_categorical(model_log_prob)
        return out
    
    def TDM_p_pred(self, log_x, t, context):
        log_x_recon = self.TDM_predict_start(log_x, t=t, context=context)
        log_model_pred = self.TDM_q_posterior(log_x_start=log_x_recon, log_x_t=log_x, t=t)
        return log_model_pred
    
    def TDM_predict_start(self, log_x_t, t, context):
        x_t = log_onehot_to_index(log_x_t)
        context = context.view(context.shape[0], -1, self.transformer_dim)
        out = self.TDM_Decoder(x_t, t, context)
        log_pred = F.log_softmax(out, dim=1)
        return log_pred

    def TDM_q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.TDM_q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.TDM_q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def TDM_q_pred(self, log_x_start, t):
        TDM_log_cumprod_alpha_t = extract(self.TDM_log_cumprod_alpha, t, log_x_start.shape)
        TDM_log_1_min_cumprod_alpha = extract(self.TDM_log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + TDM_log_cumprod_alpha_t,
            TDM_log_1_min_cumprod_alpha - np.log(self.num_classes)
        )

        return log_probs
    
    def TDM_q_pred_one_timestep(self, log_x_t, t):
        TDM_log_alpha_t = extract(self.TDM_log_alpha, t, log_x_t.shape)
        TDM_log_1_min_alpha_t = extract(self.TDM_log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + TDM_log_alpha_t,
            TDM_log_1_min_alpha_t - np.log(self.num_classes)
        )

        return log_probs

    @torch.no_grad()
    def TDM_log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample
    
    @torch.no_grad()
    def DiffTSR_sample(self, input_image, general=True):
        lq_image = np.asarray(input_image)
        lq_image = lq_image / 255.0
        lq_image = np.ascontiguousarray(lq_image)
        lq_image = transforms.ToTensor()(lq_image).float()
        lq_image = lq_image.unsqueeze(0).to(self.device)

        c_t = self.Text_Prediction.predict(lq_image)
        c_t = c_t.to(self.device)

        z_LR = self.VAE_model.encode(lq_image)
        z_LR = z_LR.sample()
        b = z_LR.shape[0]
        noise = torch.randn_like(z_LR, device=z_LR.device)
        t = torch.full((b,), 999, device=self.device, dtype=torch.long)
        z_t = self.IDM_q_sample(x_start=z_LR, t=t, noise=noise)

        timesteps = self.IDM_ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = tqdm(time_range, desc='DiffTSR Sampling', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            Icond_t, Ccond_t = self.MoM_module(c_t, z_LR, z_t, ts)
            z_t, _ = self.IDM_p_sample(z_t, z_LR, Ccond_t, ts, index=index)
            ts = torch.full((b,), index, device=self.device, dtype=torch.long)
            if not general:
                log_c_t = index_to_log_onehot(c_t, self.num_classes)
                log_c_t = self.TDM_p_sample(log_c_t, ts, context=Icond_t)
                c_t = log_onehot_to_index(log_c_t)

        img = 1. / self.scale_factor * z_t
        img = self.VAE_model.decode(img)
        img = img.detach().cpu()
        img = torch.clamp(img, 0., 1.)*255
        img = img.numpy().astype(np.uint8)[0]
        img = np.transpose(img, (1, 2, 0))

        return img