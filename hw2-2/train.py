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

hidden_size = 400
n_layers = 3
clip = 50.0

teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
atten_method = "concat"

epochs = 51
batch_size = 256

PATH_model = atten_method + '_' + str(hidden_size) + '_' + str(batch_size) + "_model"

def turn_id2word(id2word, sent):
    new_sent = []
    for i in range(len(sent)):
        new_sent.append(id2word.get(sent[i]))
    return new_sent

def train(train_data_loader, target_len, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, batch_size, clip, device):
    
    for i in range(epochs):
        for batch in range(train_data_loader.data_size//batch_size):
#         for batch in range(1):
            # zero grad
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # load data
            question, answer, mask = train_data_loader.load_on_batch(batch_size * batch, batch_size * (batch + 1))
            question, answer, mask = question.to(device), answer.to(device), mask.to(device)
            
            
            # Initialize variables
            loss = 0
            print_losses = []
            n_totals = 0

            # Forward pass through encoder
            encoder_outputs, encoder_hidden = encoder(question)
            
            # Create initial decoder input (start with SOS tokens for each sentence)
            decoder_input = torch.LongTensor([[1 for _ in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            
            # Set initial decoder hidden state to the encoder's final hidden state
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            
            # Determine if we are using teacher forcing this iteration
            use_teacher_forcing = True if np.random.random() < (teacher_forcing_ratio - 0.1 * i) else False
            
            pre_sent = []
            # Forward batch of sequences through decoder one time step at a time
            if use_teacher_forcing:
                for t in range(target_len):
                    decoder_input = decoder_input.long()
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    # Teacher forcing: next input is current target
                    decoder_input = answer[t].view(1, -1)
                    # Calculate and accumulate loss
                    mask_loss, nTotal = maskNLLLoss(decoder_output, answer[t], mask[t], device)
                    loss += mask_loss
                    n_totals += nTotal
                    
                    pre_sent.append(torch.argmax(decoder_output, dim=1).cpu().numpy())
            else:
                for t in range(target_len):
                    decoder_input = decoder_input.long()
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    # No teacher forcing: next input is decoder's own current output
                    _, topi = decoder_output.topk(1)
                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                    decoder_input = decoder_input.to(device)
                    # Calculate and accumulate loss
                    mask_loss, nTotal = maskNLLLoss(decoder_output, answer[t], mask[t], device)
                    loss += mask_loss
                    n_totals += nTotal
                    
                    pre_sent.append(torch.argmax(decoder_output, dim=1).cpu().numpy())
                    
            loss.backward()
            pre_sent = np.array(pre_sent)
            
            if(batch % 10 == 0):
                print("epochs:", i, "batch:", batch)
                print("loss:", loss)
                print("ans:", turn_id2word(train_data_loader.id2word, answer[:, 0].cpu().numpy()) )
                print()
                print("pre:", turn_id2word(train_data_loader.id2word, pre_sent[:, 0]))
                print()
                print()
            
                
            # Clip gradients: gradients are modified in place
            _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

            # Adjust model weights
            encoder_optimizer.step()
            decoder_optimizer.step()
        if(i % 10 == 0):
            torch.save(encoder, "./file/" + str(i + 51) +"_encoder_no_teach")
            torch.save(decoder, "./file/" + str(i + 51) +"_decoder_no_teach")
            
            

 

def maskNLLLoss(inp, target, mask, device):
    nTotal = mask.sum()
    eposilon = 2.220446049250313e-16
    gather = torch.gather(inp, 1, target.view(-1, 1).long())
    crossEntropy = -torch.log(gather + eposilon)
    loss = (crossEntropy * mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()  
    
    
def main():   
    # cuda
    print("enviroment setup")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # load data
    train_data_loader = DataLoader(sent_len)
    
    # constructe model
#     encoder = EncoderRNN(
#         train_data_loader.input_size, 
#         hidden_size,
#         n_layers=n_layers
#     )
#     encoder = encoder.to(device)
#     decoder = LuongAttnDecoderRNN(
#         atten_method,
#         hidden_size,
#         train_data_loader.voc_size,
#         n_layers=n_layers
#     )
#     decoder = decoder.to(device)
    encoder = torch.load("./file/50_encoder")
    encoder = encoder.to(device)
    decoder = torch.load("./file/50_decoder")
    decoder = decoder.to(device)
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    
    train(train_data_loader, target_len, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, batch_size, clip, 
          device
         )

main()    



