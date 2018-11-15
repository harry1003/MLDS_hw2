from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


################################################
#             Encoder                          #
#       Input                                  #
# input_seq [ max_len, batch_size, input_size] #
#                                              #
#       Output                                 #
# output    [batch_size, max_len, hid_size]    #
# hidden    [1         , max_len, hid_size]    #
#                                              #
# Structure                                    #
#   input-> linear-> gru-> (output & hidden)   #
################################################

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        
    def forward(self, input_seq, hidden=None):
        linear = self.linear(input_seq)
        # Forward pass through GRU
        outputs, hidden = self.gru(linear, hidden)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden