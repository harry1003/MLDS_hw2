import sys
import pickle
import numpy as np
import torch
import torch.optim as optim

from data_loader import DataLoader
from model import S2S_Net, LanguageModelCriterion

batch_size= 64
hidden_size = 400
epochs = 200
sent_len = 10
teach_rate = 0.5

train_model = 1
attention = 1


model = "S2S"
PATH = model + '_'
if attention:
    PATH = PATH + "att_"
else:
    PATH = PATH + "no_att_"
PATH_model = PATH + str(epochs) + '_' +str(hidden_size) + '_' + str(sent_len) + '.pt'


def main():
    # cuda
    print("enviroment setup")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # model & optim
    if(train_model):
        # load
        print("loading training data")
        train_data_loader = DataLoader(sent_len)
        # model
        print("model consturct")
        model = S2S_Net(train_data_loader, hidden_size, sent_len, attention)
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=0.001)
        # train
        print("start training")
        train(train_data_loader, model, opt, epochs, batch_size, device, teach_rate)



def train(data_loader, model, opt, epochs, batch_size, device, teach_rate):
    model.train()
    crit = LanguageModelCriterion()
    for i in range(epochs):
        print("epochs:", i)
        ans1, ans2 = 0, 0
        for batch in range(data_loader.data_size//batch_size):
            question, answer, mask = data_loader.load_on_batch(
                batch * batch_size, (batch + 1) * batch_size
            )
            question, answer, mask = question.to(device), answer.to(device), mask.to(device)
            pre_prob, pre_sent = model(question, answer, device, teach_rate / (i + 1))
            loss = crit(pre_prob, answer, mask)
            
            # backward
            opt.zero_grad()
            loss.backward()
            
            # grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            
            # step
            opt.step()
            
            # showing things
            if(batch % 10 == 0):
                ans1 = answer[0].cpu().numpy()
                ans2 = answer[5].cpu().numpy()
                print("epochs:", i, "batch:", batch)
                show_result(data_loader.id2word, loss, ans1, ans2, pre_sent)
        if(i % 10 == 0):
            path = str(i) + PATH_model
            torch.save(model, path)
    # save model
    torch.save(model, PATH_model)

        
def show_result(id2word, loss_ep, ans1, ans2, pre_sent_e):
    print("loss:", loss_ep)
    print()
    print("ans1:", turn_id2word(id2word, ans1))
    print("pre_sent_1:", turn_id2word(id2word, pre_sent_e[:, 0]))
    print()
    print("ans2:", turn_id2word(id2word, ans2))
    print("pre_sent_2:", turn_id2word(id2word, pre_sent_e[:, 5]))
    print()
    print()

def turn_id2word(id2word, sent):
    new_sent = []
    for i in range(len(sent)):
        new_sent.append(id2word.get(sent[i]))
    return new_sent

main()