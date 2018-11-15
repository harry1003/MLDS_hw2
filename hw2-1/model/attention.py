import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim_hidden):
        super(Attention, self).__init__()
        
        # define constant
        self.dim_hidden = dim_hidden
        
        # define layers
        self.linear1 = nn.Linear(dim_hidden * 2, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, 1, bias=False)
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, decode_hid, encode_out):
        """ 
        input shapes
            decode_hid   [ batch  dim_hidden ]
            encode_out   [ batch  n_frames  dim_hidden ]
            
        mid var shape    
            decode_hid_r [ batch  n_frames  dim_hidden ]
       
        output shape
            context      [ batch  dim_hidden ]
        """ 
        batch_size, n_frames, dim_hidden = encode_out.shape
        decode_hid_r = decode_hid.unsqueeze(1).repeat(1, n_frames, 1)
        
        weight = torch.cat((encode_out, decode_hid_r), 2).view(-1, dim_hidden * 2)
        weight = self.linear1(weight)
        weight = self.tanh(weight)
        weight = self.linear2(weight)
        weight = weight.view(batch_size, n_frames)
        weight = self.soft(weight)
        
        context = torch.bmm(weight.unsqueeze(1), encode_out).squeeze(1)
        return context