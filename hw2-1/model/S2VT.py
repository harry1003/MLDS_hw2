from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import Encoder
from .decoder import Decoder


class S2VTModel(nn.Module):
    def __init__(self, vocab_size, encode_len, decode_len, dim_hidden, dim_word, n_frames=80, dim_data=4096, 
                 n_layers=1, rnn_cell='gru', rnn_dropout_p=0.45, encode_drop=0.5, decode_drop=0.3):
        super(S2VTModel, self).__init__()
        assert(encode_len >= decode_len) # for this kind of structure
        
        self.encoder = Encoder(dim_data, dim_hidden, rnn_cell, n_layers, rnn_dropout_p, encode_drop)
        self.decoder = Decoder(vocab_size, decode_len, dim_hidden, dim_word, 
                               rnn_cell, n_layers, rnn_dropout_p, decode_drop)
        
        self.dim_data = dim_data
        self.n_frames = n_frames
        self.encode_len = encode_len
        
    def forward(self, data, target, device, teach_rate=0):
        """
        input shape
            data        [ batch  n_frames   dim_data]
            target      [ batch  encode_len       ]
            
        output  shape
            pre_prob    [ batch  encode_len  dim_word]
            pre_sent    [ batch  encode_len  ]
            
        """
        # check shape
        batch_size, n_frames, dim_data = data.shape
        assert(n_frames == self.n_frames)
        assert(dim_data == self.dim_data)
        assert(target.shape[0] == batch_size) # data, target batch_size mismatch
        assert(target.shape[1] == self.encode_len)
        # check end
        # predict start
        encode_out, encode_hid = self.encoder(data)
        pre_prob, pre_sent = self.decoder(encode_out, encode_hid, target, device, teach_rate)
        return pre_prob, pre_sent

        
        
        