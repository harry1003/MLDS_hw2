
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###############################################
#             Encoder                         #
#  Input                                      #
# x      [batch_size, input_size, hiddensize] #
#                                             #
#  Output                                     #
# output [batch_size, #time step, hid_size]   #
# hidden [1,          batch_size, hid_size]   #
#                                             #
# Structure                                   #
#   input-> linear-> gru-> (output & hidden)  #
###############################################
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        output, hidden = self.gru(x, None)
        output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
        return output, hidden

#######################################################
#             Atten_Decoder                           #
#  Input                                              #
# encode_out    [batch_size, #time step, hid_size]    #
# decode_hid    [1,          batch_size, hid_size]    #
# output        [batch_size, #time step, hid_size]    #
# embedded_word [#batch      #time_step, hid_size]    #
#                                                     #
#  Output                                             #
# decode_hid    [1,          batch_size, hid_size]    #
# word_class    [#batch      #word_class         ]    #
#                                                     #
#  Structure                                          #
# (decode_hid, word)-> gru -> (decode_out, decode_hid)#
# (encode_out, decode_hid)-> weight                   #
# (encode_out, weight)-> context                      #
# (context, decode_out)-> linear -> word_class        #
#######################################################
class Attention_Decoder(nn.Module):
    def __init__(self, hidden_size, sent_len, class_num, n_layers=1, dropout=0.1):
        super(Attention_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.sent_len = sent_len
        self.class_num = class_num
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layers,
            dropout=dropout,
            )
        self.linear = nn.Linear(hidden_size * 2, class_num)
        self.log_soft = nn.LogSoftmax()
        self.soft = nn.Softmax()
    
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
        word_class = word_class.view(-1, self.class_num)
        word_class = self.log_soft(word_class)
        # output [#batch #word_class]
        decode_hid = decode_hid.view(1, -1, self.hidden_size)
        return word_class, decode_hid

    
class S2S_Net(nn.Module):
    def __init__(self, data_loader, hidden_size=400, sent_len=10, attention=True):
        super(S2S_Net, self).__init__()
        self.sent_len = sent_len
        self.hidden_size = hidden_size
        self.class_num = data_loader.voc_size
        self.attention = attention
        self.dictionary = data_loader.dictionary
        self.id2word = data_loader.id2word
        
        self.encoder = Encoder(data_loader.input_size, hidden_size)
        print("attention used")
        self.atten_decoder = Attention_Decoder(hidden_size, sent_len, data_loader.voc_size)
        self.embed = nn.Embedding(data_loader.voc_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data, target, device, teach_rate=0.5):
        # data [batch#  #time_step 250]
        # target [batch#  #sent_len]
        # mask [batch#  #sent_len]
        batch_size = len(target)
        
        # encoding state
        encode_out, encode_hid = self.encoder(data)
        # encode_out [#batch  #time   #hidden_size]
        # encode_hid [1       #batch  #hidden_size]
        
        # decoding state
        pre_sent = []
        pre_prob = []
        decode_hid = encode_hid[3, :, :].view(1, batch_size, -1)
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
            word_class, decode_hid = self.atten_decoder(encode_out, decode_hid, embedded_word)
            
            # decode_out [#batch #class_num]
            if np.random.random() < teach_rate:
                word = target[:, i].long()
            else:
                word = torch.argmax(word_class, dim=1).long()
            # save pre sentence
            pre_sent.append(torch.argmax(word_class, dim=1).cpu().numpy())
            pre_prob.append(word_class.unsqueeze(1))
        pre_prob = torch.cat(pre_prob, 1)    
        pre_sent = np.array(pre_sent)
        return pre_prob, pre_sent