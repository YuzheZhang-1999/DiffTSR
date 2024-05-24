import zhconv
import torch
from torch import nn
import torch.nn.functional as F

from model.TDM.models.transocr import Transformer
from model.TDM.utils.util import tensor2str, get_alphabet
from model.IDM.utils.alphabets import alphabet as IDM_alphabet


class Text_Prediction(nn.Module):
    def __init__(self, 
                 imgH=32,
                 imgW=256,
                 alphabet_path='./model/TDM/utils/benchmark.txt',
                 ckpt_path='./ckpt/transocr.pth'):
        super().__init__()

        self.imgH = imgH
        self.imgW = imgW
        self.alphabet_path = alphabet_path
        self.ckpt = ckpt_path

        self.alphabet = get_alphabet(self.alphabet_path)
        self.trans_ocr_model = Transformer(len(self.alphabet)).cuda()

        weight_dict = torch.load(self.ckpt)
        weight_dict = {key.replace("module.module.", "module."): value for key, value in weight_dict.items()}

        if 'state_dict' in weight_dict:
            self.trans_ocr_model.load_state_dict(weight_dict['state_dict'])
        else:
            self.trans_ocr_model = nn.DataParallel(self.trans_ocr_model)
            self.trans_ocr_model.load_state_dict(weight_dict)

        self.trans_ocr_model.eval()
        self.IDM_alphabet = IDM_alphabet
        self.max_length = 24

    def predict(self, input_image):
        max_length = self.max_length
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

        label_index = torch.zeros(batch, max_length).type(torch.LongTensor)
        for i in range(batch):
            pred = zhconv.convert(tensor2str(text_pred_list[i], self.alphabet_path),'zh-cn')
            label_index[i, :] = self.get_padding_label_from_text(pred)
        return label_index

    def get_padding_label_from_text(self, text_str):
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
                label.append(6735)
        label = torch.tensor(label)
        if len(text_str) > max_length:
            pad_label = label[:max_length]
        else:
            pad_label = F.pad(label, (0, padding_size), value=6735)
        return pad_label