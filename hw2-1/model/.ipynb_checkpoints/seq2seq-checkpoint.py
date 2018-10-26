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
#             dropout=0.5,
            )
        # use orthogonal init for GRU layer0 weights
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)

    def forward(self, x):
        output, hidden = self.gru(x, None)
        # output [batch_size, #time step, hid_size],
        # hidden [1, batch_size, hid_size]
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, class_num):
        super(Decoder, self).__init__()
        self.class_num = class_num
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
#             dropout=0.5,
            )
        # use orthogonal init for GRU layer0 weights
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        
        self.linear = nn.Linear(hidden_size, class_num)
        self.soft = nn.LogSoftmax()

    def forward(self, embedded_word, decode_hid):
        output, hidden = self.gru(embedded_word, decode_hid)
        # output [batch_size, 1, hid_size]
        x = self.linear(output)
        x = x.view(-1, self.class_num)
        output = self.soft(x)
        return output, hidden
    

class Attention_Decoder(nn.Module):
    def __init__(self, hidden_size, sent_len, class_num):
        super(Attention_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.sent_len = sent_len
        self.class_num = class_num
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            )
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, class_num)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()
        self.log_soft = nn.LogSoftmax()
        self.dropout = nn.Dropout()
    
    def forward(self, encode_out, decode_hid, embedded_word):
        decode_out, decode_hid = self.gru(embedded_word, decode_hid)
        # encode_out[#batch #time_step #hidden_size]
        # decode_hid[1 #batch #hidden_size]
        # decode_out [#batch 1 #hidden_size]
        
        decode_hid = decode_hid.view(-1, self.hidden_size, 1)
        weight = torch.bmm(encode_out, decode_hid)
        weight = self.soft(weight)
        # weight[#batch #time_step 1]
        encode_out = encode_out.permute(0, 2, 1)
        # encode_out [#batch #hidden_size #time_step]
        context = torch.bmm(encode_out, weight)
        context = context.view(-1, 1, self.hidden_size)
        # context [#batch 1 #hidden_size]
        combine = torch.cat((context, decode_out), 2)
        #combine [#batch 1 #hid * 2]
        word_class = self.linear(combine)
        word_class = self.relu(word_class)
        word_class = self.dropout(word_class)
        word_class = self.linear2(word_class)
        word_class = word_class.view(-1, self.class_num)
        word_class = self.log_soft(word_class)
        # output [#batch #word_class]
        decode_hid = decode_hid.view(1, -1, self.hidden_size)
        return word_class, decode_hid
    
# class Attention(nn.Module):
#     def __init__(self, hidden_size, sent_len, class_num):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.sent_len = sent_len
#         self.class_num = class_num
#         self.linear = nn.Linear(hidden_size * 2, hidden_size)
#         self.relu = nn.ReLU()
#         self.soft = nn.Softmax()
    
#     def forward(self, encode_out, decode_hid, embedded_word):
#         # encode_out[#batch #time_step #hidden_size]
#         # decode_hid[1 #batch #hidden_size]
#         # decode_out [#batch 1 #hidden_size]
#         decode_hid = decode_hid.view(-1, self.hidden_size, 1)
#         weight = torch.bmm(encode_out, decode_hid)
#         weight = self.soft(weight)
#         # weight[#batch #time_step 1]
#         encode_out = encode_out.permute(0, 2, 1)
#         # encode_out [#batch #hidden_size #time_step]
#         context = torch.bmm(encode_out, weight)
#         context = context.view(-1, 1, self.hidden_size)
#         # context [#batch 1 #hidden_size]
#         combine = torch.cat((context, embedded_word), 2)
#         #combine [#batch 1 #hid * 2]
#         atten_out = self.linear(combine)
#         atten_out = self.relu(atten_out)
#         # output [#batch #word_class]
#         return atten_out
    
    
class S2S_Net(nn.Module):
    def __init__(self, data_loader, hidden_size=400, sent_len=10, attention=True):
        super(S2S_Net, self).__init__()
        self.sent_len = sent_len
        self.hidden_size = hidden_size
        self.class_num = data_loader.voc_num
        self.attention = attention
        
        self.encoder = Encoder(data_loader.input_size, hidden_size)
        self.decoder = Decoder(hidden_size, data_loader.voc_num)
        self.embed = nn.Embedding(data_loader.voc_num, hidden_size)
        self.dropout = nn.Dropout()
        if attention:
            print("attention used")
            self.atten_decoder = Attention_Decoder(hidden_size, sent_len, data_loader.voc_num)
#             self.atten = Attention(hidden_size, sent_len, data_loader.voc_num)
        
    def forward(self, data, target, mask, device, teach_rate=0.5):
        # data [batch#  #time_step 4096]
        # target [batch#  #sent_len]
        # mask [batch#  #sent_len]
        batch_size = len(mask)
        
        # encoding state
        encode_out, encode_hid = self.encoder(data)
        # encode_out [#batch  #time   #hidden_size]
        # encode_hid [1       #batch  #hidden_size]
        
        # decoding state
        losses = 0
        pre_sent = []
        
        decode_hid = encode_hid
        # decode_hid [1  #batch  #hidden_size]
        word = torch.LongTensor([1 for _ in range(batch_size)]).to(device)
        # word [#batch]
        
        for i in range(self.sent_len):
            embedded_word = self.embed(word)
            embedded_word = self.dropout(embedded_word)
            # embedded_word [#batch  #hidden_size]
            embedded_word = embedded_word.view(batch_size, 1, self.hidden_size)
            # embedded_word [#batch  1  #hidden_size]
            
            # method1
            if(self.attention):
                word_class, decode_hid = self.atten_decoder(encode_out, decode_hid, embedded_word)
            else:
                word_class, decode_hid = self.decoder(embedded_word, decode_hid)

            # method2
#             decode_in = embedded_word
#             if(self.attention):
#                 decode_in = self.atten(encode_out, decode_hid, embedded_word)
#             word_class, decode_hid = self.decoder(decode_in, decode_hid)
            
            # decode_out [#batch #class_num]
            if np.random.random() < teach_rate:
                word = target[:, i].long()
            else:
                word = torch.argmax(word_class, dim=1).long()
            # word [#batch]
            if mask[:, i].sum() != 0:
                loss = self.maskNLLLoss(word_class, target[:, i], mask[:, i], device)
                losses += loss
            # save pre sentence
            pre_sent.append(torch.argmax(word_class, dim=1).cpu().numpy())
        pre_sent = np.array(pre_sent)
        return losses, pre_sent
            

    def maskNLLLoss(self, pre, tar, mask, device):
        # pre [#batch #class_num]
        # tar [#class]
        loss_function = nn.NLLLoss()
        loss = loss_function(pre, tar.long())
        loss = loss.to(device)
        return loss
    
