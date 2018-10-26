from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            )
        # use orthogonal init for GRU layer0 weights
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)

    def forward(self, inp, hidden):
        output, hidden = self.gru(inp, hidden)
        # output [batch_size, #time step, hid_size]
        return output, hidden
    
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, class_num):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.1,
            )
        # use orthogonal init for GRU layer0 weights
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        
        self.linear2 = nn.Linear(hidden_size, class_num)
        self.soft = nn.LogSoftmax()
        self.class_num = class_num

    def forward(self, inp, hidden):
        x = self.linear1(inp)
        x = self.relu(x)
        gru_out, gru_hid = self.gru(x, hidden)
        # output [batch_size, #time step, hid_size]
        x = self.linear2(gru_out)
        x = self.soft(x)
        output = x.view(-1, self.class_num)
        return output, gru_hid
        

class Attention(nn.Module):
    def __init__(self, hidden_size, class_num):
        super(Attention, self).__init__()


class S2VT_Net(nn.Module):
    def __init__(self, data_loader, hidden_size, sent_len=10):
        super(S2VT_Net, self).__init__()
        self.input_time_step = data_loader.time_step
        self.hidden_size = hidden_size
        self.sent_len = sent_len
        self.encoder = Encoder(data_loader.input_size, hidden_size)
        self.decoder = Decoder(hidden_size, data_loader.voc_num)
        self.embed = nn.Embedding(data_loader.voc_num, hidden_size)
        

    def forward(self, data, target, mask, device, sent_len, teach_rate=0):
        # data.shape =   [batch#  #time_step 4096]
        # target.shape = [batch#  max_len#]
        # mask.shape =   [batch#  max_len#]
        batch_size = len(mask)
        
        # encoding stage
        encode_hid_all = []
        encode_hid = torch.zeros(1, batch_size, self.hidden_size).to(device)
        decode_hid = torch.zeros(1, batch_size, self.hidden_size).to(device)
        for i in range(self.input_time_step):
            # data_per_step [#batch, 1, 4096]
            data_per_step = data[:, i, :]
            data_per_step = data_per_step.view(batch_size, 1, 4096)
            encode_out, encode_hid = self.encoder(data_per_step, encode_hid)
            # encode_out [#batch, 1, #hid]
            # encode_hid [1, #batch, #hid]
            decode_in_half = encode_hid.view(batch_size, 1, self.hidden_size)
            zero = torch.zeros(batch_size, 1, self.hidden_size).to(device)
            decode_in = torch.cat((decode_in_half, zero), 2)
            #decode_in [#batch, 1, 2 * #hid]
            decode_out, decode_hid = self.decoder(decode_in, decode_hid)
            # decode_out [#batch, #class_num]
            # decode_hid [1, #batch, #hid]  
        
        
        # decodint stage
        losses = 0
        sent = []
        sent2 = []
        word = torch.ones(batch_size, dtype=torch.long).to(device)
        # word[#batch, 1]
        for i in range(sent_len):
            zero = torch.zeros(batch_size, 1, 4096).to(device)
            encode_out, encode_hid = self.encoder(zero, encode_hid)
            # encode_out [#batch, 1, #hid]
            # encode_hid [1, #batch, #hid]
            embedded_word = self.embed(word)
            embedded_word = embedded_word.view(batch_size, 1, self.hidden_size)
            # embedded_word [#batch, 1, #hidden_size]
            decode_in_half = encode_hid.view(batch_size, 1, self.hidden_size)
            decode_in = torch.cat((decode_in_half, embedded_word), 2)
            # decode_in [#batch, 1, 2 * #hid]
            decode_out, decode_hid = self.decoder(decode_in, decode_hid)
            # decode_out [#batch, #class_num]
            # decode_hid [1, #batch, #hid] 
            if np.random.random() < teach_rate:
                word = target[:, i].long()
            else:
                word = torch.argmax(decode_out, dim=1).long()
            # word[#batch, 1]
            loss = self.maskNLLLoss(decode_out, target[:, i], mask[:, i], device)
            losses += loss
            # get the predict sentence
            sent.append(torch.argmax(decode_out, dim=1)[0].cpu().numpy())
            sent2.append(torch.argmax(decode_out, dim=1)[1].cpu().numpy())
        return losses, sent, sent2
            
    def maskNLLLoss(self, pre, tar, mask, device):
        # pre [#batch #vac_num]
        # tar [#class]
        loss_function = nn.NLLLoss()
        loss = loss_function(pre, tar.long())
        loss = loss.to(device)
        return loss
    