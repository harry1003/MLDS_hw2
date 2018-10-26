import numpy as np
import torch
import json


def load_data(PATH):
    # load file name
    id_path = PATH + "/id.txt"
    file_name = []
    file = open(id_path, 'rt')
    for line in file:
        file_name.append(line[:-1])
    file.close
    # load data
    data = []
#     # temp
#     file_name = file_name[:10]
#     # temp
    for name in file_name:
        data_path = PATH + "/feat/" + name + ".npy"
        d = np.load(data_path)
        data.append(d)
    return data, file_name


def load_label(PATH):
    label_path = PATH + '.json'
    with open(label_path, 'r') as reader:
        label = json.loads(reader.read())
    return label


def load_everything(PATH, mode):
    if mode == "train":
        PATH = PATH + "training_"
    elif mode == "test":
        PATH = PATH + "testing_"
    else:
        print("mode must be 'train' or 'test'")
    data, file_name = load_data(PATH + 'data')
    label = load_label(PATH + 'label')
    return data, label, file_name
    # where we can read label by label[id_num][caption][# of lines]


word_count_threshold = 3


class data_loader():
    def __init__(self, data, label, sent_len):
        super(data_loader, self).__init__()
        self.data = data
        self.label = label
        word2id, id2word, max_lenth, voc_num = preprocess(label)
        self.word2id = word2id
        self.id2word = id2word
        self.max_lenth = max_lenth
        self.sent_len = sent_len
        self.voc_num = voc_num
        self.input_size = data.shape[2]
        self.time_step = data.shape[1]
        self.data_size = len(data)

    def load_on_batch(self, start, end, epochs):
        data = self.data[start:end]
        target = []
        mask = []
        for i in range(start, end):
            num_sent_have = len(self.label[i]['caption'])
            sent_take = epochs % num_sent_have
            sent = self.label[i]['caption'][sent_take]
            v, m = self.turn_sent_to_vec(sent)
            target.append(v)
            mask.append(m)
        target = np.array(target)
        mask = np.array(mask)
        data = torch.Tensor(data)
        target = torch.Tensor(target)
        mask = torch.Tensor(mask)
        return data, target, mask

    def turn_sent_to_vec(self, sentence):
        vec = []
        mask = []
        sentence = sentence.split('.')[0]
        sentence = sentence.split(' ')
        for i in range(self.sent_len):
            if i < len(sentence):
                mask.append(1)
                w = sentence[i]
                id = self.word2id.get(w)
                if(id):
                    vec.append(id)
                else:
                    vec.append(self.word2id.get('<unk>'))
            elif i == len(sentence):
                mask.append(1)
                vec.append(self.word2id.get('<eos>'))
            else:
                mask.append(0)
                vec.append(self.word2id.get('<pad>'))
        vec = np.array(vec)
        mask = np.array(mask)
        return vec, mask
    
    def turn_vec_to_sent(self, vec):
        temp = []
        for i in range(len(vec)):
            temp.append(self.id2word[int(vec[i])])
        return temp
    
    def turn_tar_to_sent(self, vec):
        temp = ""
        for i in range(len(vec)):
            if vec[i] == 2:
                temp = temp[:-1]
                temp = temp + '.'
                return temp
            elif vec[i] == 0 or vec[i] == 3 or vec[i] == 1:
                temp = temp + ''
            else:
                temp = temp + self.id2word[vec[i]] + ' '
        return temp

    
def preprocess(label):
    nsents = 0
    word_counts = {}
    max_lenth = 0
    for video in label:
        for sent in video['caption']:
            nsents += 1
            split = sent.split(' ')
            if len(split) > max_lenth:
                max_lenth = len(split)
            for w in split:
                word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))
    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'
    
    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3
    
    # get the index and content, that is (idx,word)
    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w
    return wordtoix, ixtoword, max_lenth, len(vocab)
