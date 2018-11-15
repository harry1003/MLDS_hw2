from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from data_loader import DataLoader
from model import EncoderRNN, LuongAttnDecoderRNN

# constant
sent_len = 10
target_len = 10

def turn_id2word(id2word, sent):
    new_sent = []
    for i in range(len(sent)):
        new_sent.append(id2word.get(sent[i]))
    return new_sent

def predict(test_data_loader, id2word, target_len, encoder, decoder, device):
    
    encoder.eval()
    decoder.eval()
    
    predict_size = test_data_loader.data_size
    batch_size = 100
    
    # predict file
    file = open("output.txt", "w")
    
    for i in range(predict_size//batch_size):
        # load data
        question, _, _ = test_data_loader.load_on_batch(i * batch_size, (i + 1) * batch_size)
        question = question.to(device)


        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(question)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[1 for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]


        pre_sent = []
        for t in range(target_len):
            decoder_input = decoder_input.long()
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(100)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss                    
            pre_sent.append(torch.argmax(decoder_output, dim=1).cpu().numpy())
        pre_sent = np.array(pre_sent)    
        pre_sent = np.transpose(pre_sent)

        for i in range(100):
            sent = turn_id2word(id2word, pre_sent[i])
            new_sent = out_process(sent)
            new_sent = new_sent + '\n'
            file.write(new_sent)
    file.close()
   

def out_process(sent):
    new_sent = ''
    temp_word = 0
    for i in range(len(sent)):
        if sent[i] == '<eos>':
            return new_sent
        if temp_word == sent[i] or sent[i]=='<pad>' or sent[i]=='<unk>':
            temp_word = sent[i]
        else:
            temp_word = sent[i]
            new_sent = new_sent + sent[i] + ' '
    return new_sent
            
    
def main():   
    # cuda
    print("enviroment setup")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    path = "./data_loader/test_input.txt"
    # load data
    test_data_loader = DataLoader(sent_len, path, path)
    train_data_loader = DataLoader(sent_len)
    id2word = train_data_loader.id2word
    # constructe model
    encoder = torch.load("./file/50_encoder")
    encoder = encoder.to(device)
    decoder = torch.load("./file/50_decoder")
    decoder = decoder.to(device)
    
    predict(test_data_loader, id2word, target_len, encoder, decoder, device)

main()    


def turn_id2word(id2word, sent):
    new_sent = []
    for i in range(len(sent)):
        new_sent.append(id2word.get(sent[i]))
    return new_sent