import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, decode_len, dim_hidden, dim_word,
                 rnn_cell, n_layers, rnn_dropout_p, linear_dropout_p=0.35):
        super(Decoder, self).__init__()
        # layer type setup
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
            
        # define constant
        self.vocab_size = vocab_size
        self.decode_len = decode_len
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.rnn_type = rnn_cell
        
        # define layer
        self.embed = nn.Embedding(vocab_size, dim_word) 
        self.dropout_embed = nn.Dropout(linear_dropout_p) 
        self.rnn = self.rnn_cell(
            dim_hidden + dim_word, 
            dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)
        self.attention = Attention(dim_hidden)
        self.linear_h2v = nn.Linear(dim_hidden, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # init weight
        self._init_linear()
        self._init_rnn_weight()
        
    def _init_linear(self):
        nn.init.xavier_normal_(self.linear_h2v.weight)
        
    def _init_rnn_weight(self):
        for weight in self.rnn.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal(weight.data)
        
    def forward(self, encode_out, encode_hid, target, device, teach_rate):
        """ 
        input and output shapes
            encode_out   [ batch  n_frames   dim_hidden ]
            encode_hid   [ 1      batch      dim_hidden ]
            target       [ batch  encode_len ]
        
        mid var shape
            current_word [ batch  dim_word ]
            target_embed [ batch  encode_len  dim_word  ]
            decode_out   [ batch  1           dim_hidden ]
            decoee_hid   [ 1      batch       dim_hidden ]
            word_class   [ batch  vocab_size  ]
            log_prob     [ batch  vacab_size  ]
            sent         [ batch  1           ]
        
        output  shape
            pre_prob     [ batch        encode_len  dim_word]
            pre_sent     [ encode_len   batch     ]
            
        """ 
        batch_size, n_frames, dim_hidden = encode_out.shape
        
        # init hidden state
        decode_hid = encode_hid
        
        # output
        pre_prob = []
        pre_sent = []
        
        # set up
        bos = torch.ones(batch_size).long().view(batch_size, 1).to(device)
        target = torch.cat((bos, target.long()), dim=1)
        target_embed = self.embed(target)
        word = bos
        # start decode
        self.rnn.flatten_parameters() # why?
        for i in range(self.decode_len):
            if teach_rate >= np.random.random():
                # teach mode
                current_word = target_embed[:, i, :]
            else:
                # self mode
                current_word = self.embed(word).squeeze(1)
            if self.rnn_type == "lstm":
                atten_in = decode_hid[0].view(1, batch_size, self.dim_hidden)
            else:
                atten_in = decode_hid.view(1, batch_size, self.dim_hidden)
            context = self.attention(atten_in.squeeze(0), encode_out)
            decode_in = torch.cat((current_word, context), dim=1)
            decode_in = self.dropout_embed(decode_in).unsqueeze(1)
            decode_out, decode_hid = self.rnn(decode_in, decode_hid)
                
            word_class = decode_out.squeeze(1)
            word_class = self.linear_h2v(word_class)
            
            log_prob = self.log_softmax(word_class)
            pre_prob.append(log_prob.unsqueeze(1))
            
            word = torch.argmax(log_prob, dim=1)
            pre_sent.append(word.cpu().numpy())
              
        pre_prob = torch.cat(pre_prob, 1)
        pre_sent = np.array(pre_sent)

        return pre_prob, pre_sent