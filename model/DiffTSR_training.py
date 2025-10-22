import torch, math
import numpy as np
from functools import partial
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.optim.lr_scheduler import LambdaLR

from model.IDM.utils.util import instantiate_from_config, default
from model.IDM.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor

from model.TDM.models.diffusion_multinomial import log_categorical, sum_except_batch
from model.TDM.models.diffusion_multinomial import extract, log_add_exp, cosine_beta_schedule
from model.TDM.models.diffusion_multinomial import log_1_min_a, index_to_log_onehot, log_onehot_to_index

from model.modules.losses import Trans_OCR_loss


class DiffTSR_training_pipline(pl.LightningModule):
    def __init__(self,
                 max_length = 24,
                 num_classes = 6736,
                 transformer_dim = 768,
                 scale_factor = 0.18215,
                 learning_rate = 1.0e-05,
                 IDM_Unet_config = None,
                 TDM_Decoder_config = None,
                 MoM_module_config = None,
                 VAE_model_config = None,
                 VAE_pretrained_ckpt = None,
                 IDM_pretrained_ckpt = None,
                 TDM_pretrained_ckpt = None,
                 use_scheduler = None,
                 lr_image_key = 'lq_image',
                 hr_image_key = 'hq_image',
                 text_label_key = 'text_label',
                 ocr_label_key = 'label_ocr'):
        super().__init__()
        
        self.max_length = max_length
        self.num_classes = num_classes
        self.transformer_dim = transformer_dim
        self.scale_factor = scale_factor
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler
        
        self.lr_image_key = lr_image_key
        self.hr_image_key = hr_image_key
        self.text_label_key = text_label_key
        self.ocr_label_key = ocr_label_key

        self.VAE_model = instantiate_from_config(VAE_model_config).to(self.device).eval()
        self.IDM_Unet = instantiate_from_config(IDM_Unet_config).to(self.device).eval()
        self.TDM_Decoder = instantiate_from_config(TDM_Decoder_config).to(self.device).eval()
        self.MoM_module = instantiate_from_config(MoM_module_config).to(self.device)

        # load pretrained IDM Unet
        VAE_state_dict = torch.load(VAE_pretrained_ckpt, map_location=self.device)
        self.VAE_model.load_state_dict(VAE_state_dict['state_dict'], strict=True)
        
        self.init_from_ckpt(self.IDM_Unet, IDM_pretrained_ckpt, 'model.diffusion_model.', 
                            ['first_stage_model', 'hybrid_concat_cond_stage_model', 'pixel_loss',
                             'hybrid_cross_attn_cond_stage_model'])
            
        self.init_from_ckpt(self.TDM_Decoder, TDM_pretrained_ckpt, 'denoise_fn.diffusion_model.',
                            ['cond_stage_model'])

        self.init_MoM_transformer_from_ckpt(self.MoM_module, IDM_pretrained_ckpt,
                                            ['first_stage_model', 'hybrid_concat_cond_stage_model', 'model.diffusion_model'])
        
        self.IDM_ddpm_schedule_init(timesteps=1000, schedule="linear", 
                                    linear_start=0.0015, linear_end=0.0205, cosine_s=8e-3)

        self.TDM_ddpm_schedule_init(timesteps=1000)
            
        ## define loss functions
        self.IDM_recognize_loss_metric = Trans_OCR_loss(0.02, 32, 256, 
                                                 'model/TDM/utils/benchmark.txt', 
                                                 'train/ckpt/others/transocr.pth')

        self.VAE_model.train = disabled_train
        for param in self.VAE_model.parameters():
            param.requires_grad = False
        self.IDM_Unet.train = disabled_train
        self.TDM_Decoder.train = disabled_train
        self.IDM_recognize_loss_metric.train = disabled_train
        for param in self.IDM_recognize_loss_metric.parameters():
            param.requires_grad = False

    def init_from_ckpt(self, model, path, delet_prefix=None, ignore_keys=None):
        model_weight = torch.load(path, map_location="cpu")
        if "state_dict" in list(model_weight.keys()):
            model_weight = model_weight["state_dict"]
        
        if ignore_keys is not None:
            keys = list(model_weight.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        del model_weight[k]
        if delet_prefix is not None:
            keys = list(model_weight.keys())
            for k in keys:
                if delet_prefix != '' and k.startswith(delet_prefix):
                    new_k = k[len(delet_prefix):]
                    model_weight[new_k] = model_weight.pop(k)

        missing, unexpected = model.load_state_dict(model_weight, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def init_MoM_transformer_from_ckpt(self, model, path, ignore_keys=None):
        model_weight = torch.load(path, map_location="cpu")
        if "state_dict" in model_weight:
            model_weight = model_weight["state_dict"]

        prefix = "hybrid_cross_attn_cond_stage_model."
        filtered_weight = {}
        for k, v in model_weight.items():
            if k.startswith(prefix):
                new_k = "Transformer." + k[len(prefix):]
                filtered_weight[new_k] = v

        if ignore_keys is not None:
            keys = list(filtered_weight.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print(f"Deleting key {k} from state_dict.")
                        del filtered_weight[k]

        missing, unexpected = model.load_state_dict(filtered_weight, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing[:20]} ...")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected[:20]} ...")

    def IDM_ddpm_schedule_init(self, timesteps, schedule, linear_start, linear_end, cosine_s):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        IDM_betas = make_beta_schedule(schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        IDM_alphas = 1. - IDM_betas
        IDM_alphas_cumprod = np.cumprod(IDM_alphas, axis=0)
        IDM_alphas_cumprod_prev = np.append(1., IDM_alphas_cumprod[:-1])

        timesteps, = IDM_betas.shape
        self.IDM_ddpm_num_timesteps = int(timesteps)

        assert IDM_alphas_cumprod.shape[0] == self.IDM_ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('IDM_betas', to_torch(IDM_betas))
        self.register_buffer('IDM_alphas_cumprod', to_torch(IDM_alphas_cumprod))
        self.register_buffer('IDM_alphas_cumprod_prev', to_torch(IDM_alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('IDM_sqrt_alphas_cumprod', to_torch(np.sqrt(IDM_alphas_cumprod)))
        self.register_buffer('IDM_sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - IDM_alphas_cumprod)))
        self.register_buffer('IDM_log_one_minus_alphas_cumprod', to_torch(np.log(1. - IDM_alphas_cumprod)))
        self.register_buffer('IDM_sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / IDM_alphas_cumprod)))
        self.register_buffer('IDM_sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / IDM_alphas_cumprod - 1)))

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

    def IDM_q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.IDM_sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.IDM_sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def IDM_predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.IDM_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.IDM_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

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
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.TDM_q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        unnormed_logprobs = log_EV_qxtmin_x0 + self.TDM_q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def TDM_kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.TDM_q_pred(log_x_start, t=(self.IDM_ddpm_num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.TDM_multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)
    
    def TDM_q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.TDM_q_pred(log_x_start, t)
        log_sample = self.TDM_log_sample_categorical(log_EV_qxt_x0)
        return log_sample
    
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
    
    def TDM_multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl
    
    @torch.no_grad()
    def get_input(self, batch, lr_image_key, hr_image_key, text_label_key, ocr_label_key):
        lr_image = batch[lr_image_key]
        hr_image = batch[hr_image_key]
        text_label = batch[text_label_key]
        ocr_label = batch[ocr_label_key]

        lr_image = rearrange(lr_image, 'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()
        hr_image = rearrange(hr_image, 'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()

        lr_latent = self.VAE_model.encode(lr_image).sample().detach()*self.scale_factor
        hr_latent = self.VAE_model.encode(hr_image).sample().detach()*self.scale_factor

        return lr_image, hr_image, lr_latent, hr_latent, text_label, ocr_label


    def forward(self, lr_latent, hr_latent, text_label, ocr_label):
        t = torch.randint(0, self.IDM_ddpm_num_timesteps, (hr_latent.shape[0],), device=self.device).long()
        
        ## for IDM add noise
        IDM_noise = torch.randn_like(hr_latent)
        IDM_x_noisy = self.IDM_q_sample(x_start=hr_latent, t=t, noise=IDM_noise)
        
        ## for TDM add noise
        TDM_log_x_start = index_to_log_onehot(text_label, self.num_classes)
        TDM_log_x_noisy = self.TDM_q_sample(log_x_start=TDM_log_x_start, t=t)
        TDM_log_true_prob = self.TDM_q_posterior(log_x_start=TDM_log_x_start, log_x_t=TDM_log_x_noisy, t=t)

        ## for MoM get condition
        I_cond, C_cond = self.MoM_module(c_t=text_label, z_lr=lr_latent, z_t=IDM_x_noisy, t=t)

        ## IDM predict
        IDM_model_predict = self.IDM_Unet(x=torch.cat([IDM_x_noisy, lr_latent], dim=1), timesteps=t, context=C_cond)

        ## TDM predict
        TDM_log_model_predict = self.TDM_p_pred(log_x=TDM_log_x_noisy, t=t, context=I_cond)

        ## calculate IDM loss
        IDM_denoise_loss = (IDM_noise - IDM_model_predict).abs().mean(dim=[1, 2, 3])
        IDM_predict_x0_latent = self.IDM_predict_start_from_noise(IDM_x_noisy, t=t, noise=IDM_model_predict)
        IMD_predict_x0 = self.VAE_model.decode(1. / self.scale_factor * IDM_predict_x0_latent)
        IDM_recognize_loss = self.IDM_recognize_loss_metric(IMD_predict_x0, ocr_label)
        IDM_loss = IDM_denoise_loss + IDM_recognize_loss * 1.0
        IDM_loss = IDM_loss.mean()

        ## calculate TDM loss
        TDM_kl = self.TDM_multinomial_kl(TDM_log_true_prob, TDM_log_model_predict)
        TDM_kl = sum_except_batch(TDM_kl)
        decoder_nll = -log_categorical(TDM_log_x_start, TDM_log_model_predict)
        decoder_nll = sum_except_batch(decoder_nll)
        mask = (t == torch.zeros_like(t)).float()
        TDM_kl = mask * decoder_nll + (1. - mask) * TDM_kl
        kl_prior = self.TDM_kl_prior(TDM_log_x_start)
        vb_loss = TDM_kl / (torch.ones_like(t).float() / self.IDM_ddpm_num_timesteps) + kl_prior
        TDM_loss = 0.1 * vb_loss.sum() / (math.log(2.) * hr_latent.shape[0] * self.max_length)
    
        ##
        total_loss = IDM_loss + TDM_loss

        loss_dict = {
            'total_loss': total_loss,
            'IDM_loss': IDM_loss,
            'TDM_loss': TDM_loss
        }

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        _, _, lr_latent, hr_latent, text_label, ocr_label = self.get_input(batch,
                                                                           lr_image_key=self.lr_image_key,
                                                                           hr_image_key=self.hr_image_key,
                                                                           text_label_key=self.text_label_key,
                                                                           ocr_label_key=self.ocr_label_key)
        loss, loss_dict = self.forward(lr_latent, hr_latent, text_label, ocr_label)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, _, lr_latent, hr_latent, text_label, ocr_label = self.get_input(batch,
                                                                           lr_image_key=self.lr_image_key,
                                                                           hr_image_key=self.hr_image_key,
                                                                           text_label_key=self.text_label_key,
                                                                           ocr_label_key=self.ocr_label_key)
        _, loss_dict = self.forward(lr_latent, hr_latent, text_label, ocr_label)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.MoM_module.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
    

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
