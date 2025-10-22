import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?

import zhconv
from einops import rearrange
from model.TDM.models.transocr import Transformer
from model.IDM.utils.alphabets import alphabet as IDM_alphabet
from model.TDM.utils.util import tensor2str, get_alphabet, converter


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", vgg_lpips_local_ckpt_path=None, vgg16_local_ckpt_path=None):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        # self.perceptual_loss = LPIPS(vgg_lpips_local_ckpt_path=vgg_lpips_local_ckpt_path, vgg16_local_ckpt_path = vgg16_local_ckpt_path).eval()
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None, return_dic=False):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.mean(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss) / nll_loss.shape[0]
        if self.kl_weight>0:
            kl_loss = posteriors.kl()
            kl_loss = torch.mean(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    # assert not self.training
                    d_weight = torch.tensor(1.0) * self.discriminator_weight
            else:
                # d_weight = torch.tensor(0.0)
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if self.kl_weight>0:
                loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
                log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                       "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                       "{}/rec_loss".format(split): rec_loss.detach().mean(),
                       "{}/d_weight".format(split): d_weight.detach(),
                       "{}/disc_factor".format(split): torch.tensor(disc_factor),
                       "{}/g_loss".format(split): g_loss.detach().mean(),
                       }
                if return_dic:
                    loss_dic = {}
                    loss_dic['total_loss'] = loss.clone().detach().mean()
                    loss_dic['logvar'] = self.logvar.detach()
                    loss_dic['kl_loss'] = kl_loss.detach().mean()
                    loss_dic['nll_loss'] = nll_loss.detach().mean()
                    loss_dic['rec_loss'] = rec_loss.detach().mean()
                    loss_dic['d_weight'] = d_weight.detach()
                    loss_dic['disc_factor'] = torch.tensor(disc_factor)
                    loss_dic['g_loss'] = g_loss.detach().mean()
            else:
                loss = weighted_nll_loss + d_weight * disc_factor * g_loss
                log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                       "{}/nll_loss".format(split): nll_loss.detach().mean(),
                       "{}/rec_loss".format(split): rec_loss.detach().mean(),
                       "{}/d_weight".format(split): d_weight.detach(),
                       "{}/disc_factor".format(split): torch.tensor(disc_factor),
                       "{}/g_loss".format(split): g_loss.detach().mean(),
                       }
                if return_dic:
                    loss_dic = {}
                    loss_dic["{}/total_loss".format(split)] = loss.clone().detach().mean()
                    loss_dic["{}/logvar".format(split)] = self.logvar.detach()
                    loss_dic['nll_loss'.format(split)] = nll_loss.detach().mean()
                    loss_dic['rec_loss'.format(split)] = rec_loss.detach().mean()
                    loss_dic['d_weight'.format(split)] = d_weight.detach()
                    loss_dic['disc_factor'.format(split)] = torch.tensor(disc_factor)
                    loss_dic['g_loss'.format(split)] = g_loss.detach().mean()

            if return_dic:
                return loss, log, loss_dic
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }

            if return_dic:
                loss_dic = {}
                loss_dic["{}/disc_loss".format(split)] = d_loss.clone().detach().mean()
                loss_dic["{}/logits_real".format(split)] = logits_real.detach().mean()
                loss_dic["{}/logits_fake".format(split)] = logits_fake.detach().mean()
                return d_loss, log, loss_dic

            return d_loss, log


class Trans_OCR_loss(nn.Module):
    def __init__(self, 
                 loss_weight=0.01, 
                 imgH=32, 
                 imgW=256, 
                 alphabet_path='model/TDM/utils/benchmark.txt', 
                 ckpt_path='train/ckpt/others/transocr.pth'):
        super().__init__()

        self.imgH = imgH
        self.imgW = imgW
        self.alphabet_path = alphabet_path
        self.ckpt = ckpt_path
        self.loss_weight = loss_weight

        self.alphabet = get_alphabet(self.alphabet_path)
        self.trans_ocr_model = Transformer(len(self.alphabet)).cuda()

        print('Loading Trans_OCR_loss from: ', self.ckpt)

        weight_dict = torch.load(self.ckpt)
        weight_dict = {key.replace("module.module.", "module."): value for key, value in weight_dict.items()}

        if 'state_dict' in weight_dict:
            self.trans_ocr_model.load_state_dict(weight_dict['state_dict'])
        else:
            self.trans_ocr_model = nn.DataParallel(self.trans_ocr_model)
            self.trans_ocr_model.load_state_dict(weight_dict)

        self.trans_ocr_model.eval()
        self.ocr_criterion = torch.nn.CrossEntropyLoss()

        self.IDM_alphabet = IDM_alphabet
        self.max_length = 24

    def forward(self, recon_image, gt_label):

        length, text_input, text_gt, string_label = converter(gt_label)
        recon_image = torch.nn.functional.interpolate(recon_image, size=(self.imgH, self.imgW))
        result = self.trans_ocr_model(recon_image, length, text_input)
        text_pred = result['pred']
        loss_ocr = self.ocr_criterion(text_pred, text_gt)
        return self.loss_weight*loss_ocr

    def ocr_detect(self, input_image):
        max_length = self.max_length
        input_image = rearrange(input_image, 'b h w c -> b c h w')
        input_image = torch.nn.functional.interpolate(input_image, size=(self.imgH, self.imgW))
        batch = input_image.size()[0]
        pred = torch.zeros(batch,1).long().cuda()
        image_features = None
        prob = torch.zeros(batch, max_length).float()
        for i in range(max_length):
            length_tmp = torch.zeros(batch).long().cuda() + i + 1
            result = self.trans_ocr_model(input_image, length_tmp, pred, conv_feature=image_features, test=True)
            prediction = result['pred']
            now_pred = torch.max(torch.softmax(prediction,2), 2)[1]
            prob[:,i] = torch.max(torch.softmax(prediction,2), 2)[0][:,-1]
            pred = torch.cat((pred, now_pred[:,-1].view(-1,1)), 1)
            image_features = result['conv']

        text_pred_list = []
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(self.alphabet) - 1:
                    now_pred.append(pred[i][j])
                else:
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            overall_prob = 1.0
            for j in range(len(now_pred)):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)

        ldm_label_index = torch.zeros(batch, max_length).type(torch.LongTensor)
        for i in range(batch):
            pred = zhconv.convert(tensor2str(text_pred_list[i], self.alphabet_path),'zh-cn')
            ldm_label_index[i, :] = self.get_ldm_padding_label_from_text(pred)
        return ldm_label_index, pred


    def get_ldm_padding_label_from_text(self, text_str):
        max_length = self.max_length
        padding_size = max(max_length - len(text_str), 0)
        check_text = ''
        label = []
        for i in text_str:
            index = self.IDM_alphabet.find(i)
            if index >= 0:
                check_text = check_text + i
                label.append(index)
            else:
                label.append(6735)   # 如果字符库 alphabet中没有该字， 则用 6735 代替

        label = torch.tensor(label)
        if len(text_str) > max_length:
            pad_label = label[:max_length]
        else:
            pad_label = F.pad(label, (0, padding_size), value=6735)
        return pad_label