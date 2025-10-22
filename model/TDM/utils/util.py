import torch
import collections

global alp2num_character, alphabet_character
alp2num_character = None


def get_alphabet(alpha_path):
    global alp2num_character, alphabet_character
    if alp2num_character == None:
        alphabet_character_file = open(alpha_path)
        alphabet_character = list(alphabet_character_file.read().strip())
        alphabet_character_raw = ['START', '\xad']
        for item in alphabet_character:
            alphabet_character_raw.append(item)
        alphabet_character_raw.append('END')
        alphabet_character = alphabet_character_raw
        alp2num = {}
        for index, char in enumerate(alphabet_character):
            alp2num[char] = index
        alp2num_character = alp2num

    return alphabet_character


def tensor2str(tensor, alpha_path):
    alphabet = get_alphabet(alpha_path)
    string = ""
    for i in tensor:
        if i == (len(alphabet)-1):
            continue
        string += alphabet[i]
    return string


class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        alphabet = list(alphabet)
        alphabet.append('PAD')
        self.alphabet = alphabet
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i

    def encode(self,text):
        if isinstance(text, str):
            text_raw = text
            text = []
            for char in text_raw:
                if char not in self.dict.keys():
                    text.append(self.dict['PAD'])
                else:
                    text.append(self.dict[char])
            length = len(text)

        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
            
        return torch.tensor(text, dtype=torch.int64), length

    def decode(self, texts, lengths, raw = False):
        if len(lengths) == 1:
            length = lengths[0]
            if raw:
                return ''.join([self.alphabet[i] for i in texts])
            else:
                text = []
                for i in range(length):
                    if texts[i] < self.dict['PAD']:
                        text.append(self.alphabet[texts[i]])
                return ''.join(text)
        else:
            res = []
            for i in range(len(lengths)):
                res.append(self.decode(texts[i], [lengths[i]], raw = raw))
            return res


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def converter(label):
    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            try:
                text_input[i][j + 1] = alp2num[label[i][j]]
            except:
                text_input[i][j + 1] = alp2num['~']

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i])-1):
                text_all[start + j] = alp2num['END']
            else:
                try:
                    text_all[start + j] = alp2num[label[i][j]]
                except:
                    text_all[start + j] = alp2num['~']
        start += len(label[i])

    else:
        return length, text_input, text_all, string_label
