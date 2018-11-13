#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
import numpy as np
import torch

PATH_Q = "./data_loader/question.txt"
PATH_A = "./data_loader/answer.txt"
PATH_model = "./data_loader/dictionary"
with open(PATH_model, 'rb') as file:
    dictionary = pickle.load(file)
      
word_count_threshold = 3


def readfile():
    question = open(PATH_Q, 'r')
    answer = open(PATH_A, 'r')
    q = question.readlines()
    a = answer.readlines()
    return q, a
            
    
class DataLoader():
    def __init__(self, sent_len):
        q, a = readfile()
        self.question = q
        self.answer = a
        self.sent_len = sent_len
        self.dictionary = dictionary
        print("init word_2_vec")
        word2id, id2word, voc_size, data_size = preprocess(a)
        self.word2id = word2id
        self.id2word = id2word
        self.voc_size = voc_size
        self.data_size = data_size
        self.input_size = 250
        
    def load_on_batch(self, start, end):
        q = self.question[start:end]
        a = self.answer[start:end]
        question = []
        mask = []
        answer = []
        
        for i in range(len(q)):
            v, m = self.turn_sent_to_vec(q[i])
            question.append(v)
            
        for i in range(len(q)):
            v, m = self.turn_sent_to_id(a[i])
            answer.append(v)
            mask.append(m)
            
        question = np.array(question)
        answer = np.array(answer)
        mask = np.array(mask)
        question = torch.Tensor(question)
        answer = torch.Tensor(answer)
        mask = torch.Tensor(mask)
        return question, answer, mask
    
    def turn_sent_to_vec(self, sentence):
        vec = []
        mask = []
        sentence = sentence[:-1].split(' ')
        for i in range(self.sent_len):
            if i < len(sentence):
                mask.append(1)
                w = sentence[i]
                id = self.dictionary.get(w)
                if(id is not None):
                    vec.append(id)
                else:
                    vec.append(np.ones((250))) # unknown = 1
            elif i == len(sentence):
                mask.append(1)
                vec.append(np.zeros((250))) # end = 0
            else:
                mask.append(0)
                vec.append(np.ones((250)) * 2) # blank = 2
        vec = np.array(vec)
        mask = np.array(mask)
        return vec, mask

    def turn_sent_to_id(self, sentence):
        vec = []
        mask = []
        sentence = sentence[:-1].split(' ')
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
  

def preprocess(label):
    nsents = 0
    word_counts = {}
    max_lenth = 0
    for sent in label:
        split = sent[:-1].split(' ')
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
    return wordtoix, ixtoword, len(wordtoix), len(label)

if __name__ == '__main__':
    # for testing purpose
    d = DataLoader(10)
    question, answer, mask = d.load_on_batch(0, 1)
    print(answer)
    # print(d.data_size)