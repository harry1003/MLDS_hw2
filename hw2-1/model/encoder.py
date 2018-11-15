import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, dim_data, dim_hidden, rnn_cell, n_layers, 
                 rnn_dropout_p, linear_dropout_p=0.5):
        super(Encoder, self).__init__()
        # layer type setup
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        
        # define constant
        self.dim_hidden = dim_hidden
        
        # define layer
        self.linear_d2h = nn.Linear(dim_data, dim_hidden)
        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers,
                                 batch_first=True, dropout=rnn_dropout_p)
        self.dropout = nn.Dropout(linear_dropout_p)
        
        # init weight
        self._init_linear()
        self._init_rnn_weight()
        self.hidden = None
        
    def forward(self, data):
        """ 
        input and output shapes
            data       [ batch  n_frames  dim_data   ]
            encode_out [ batch  n_frames  dim_hidden ]
            encode_hid [ 1      batch     dim_hidden ]
        """
        batch_size, n_frames, dim_data = data.shape
        data = data.view(-1, dim_data)
        data = self.linear_d2h(data)
        data = self.dropout(data)
        data = data.view(batch_size, n_frames, self.dim_hidden)
        self.rnn.flatten_parameters() #why?
        encode_out, encode_hid = self.rnn(data, self.hidden)
        return encode_out, encode_hid
    
    def _init_linear(self):
        nn.init.xavier_normal_(self.linear_d2h.weight)
    
    def _init_rnn_weight(self):
        for weight in self.rnn.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal(weight.data)