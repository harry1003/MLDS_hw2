import sys
import pickle
import numpy as np
import torch
import torch.optim as optim

from load import load_everything, data_loader
from model import S2S_Net, S2VT_Net

batch_size= 200
hidden_size = 400
epochs = 500
sent_len = 10
teach_rate = 0.5
test = 1
used_sent = 100
attention = 1
model = "S2S"

PATH = model + '_'
if attention:
    PATH = PATH + "att_"
else:
    PATH = PATH + "no_att_"
PATH_model = PATH + str(epochs) + '_' +str(hidden_size) + '_' + str(sent_len) + '_' + str(used_sent) + '.pt'
PATH_save = PATH + str(epochs) + '_' +str(hidden_size) + '_' + str(sent_len) + '_' + str(used_sent) + '.txt'


def main(data_path, output_path, train_model):
    # cuda
    print("enviroment setup")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # model & optim
    if(train_model):
        # load
        print("loading training data")
        data, label, _ = load_everything(data_path, "train")
        data = np.array(data)
        data_load = data_loader(data, label, sent_len)
        # model
        print("model consturct")
        model = S2S_Net(data_load, hidden_size, sent_len, attention)
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=0.001)
        # train
        print("start training")
        train(data_load, model, opt, epochs, batch_size, device, teach_rate)
    # eval
    if(test):
        print("eval")
        print("loading training data")
        data, _, file_name = load_everything(data_path, "test")
        evaluate(data, sent_len, file_name, device, output_path)


def train(data_loader, model, opt, epochs, batch_size, device, teach_rate):
    model.train()
    for i in range(epochs):
        print("epochs:", i)
        loss_ep, pre_sent = 0, 0
        ans = 0
        ans2 = 0
        for batch in range(data_loader.data_size//batch_size):
            data, target, mask = data_loader.load_on_batch(
                batch * batch_size, (batch + 1) * batch_size,
                i % used_sent)
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            loss, pre_sent = model(data, target, mask, device, teach_rate / (i + 1))
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_ep += loss
            ans = target[0].cpu().numpy()
            ans2 = target[1].cpu().numpy()
        av_loss = loss_ep/(data_loader.data_size//batch_size)
        print("loss:", av_loss)
    # for checking the answer
        sent = data_loader.turn_tar_to_sent(pre_sent[:, 0])
        sent2 = data_loader.turn_tar_to_sent(pre_sent[:, 1])
        ans = data_loader.turn_vec_to_sent(ans)
        ans2 = data_loader.turn_vec_to_sent(ans2)
        print()
        print("sent", sent)
        print("ans", ans)
        print()
        print("sent2", sent2)
        print("ans2", ans2)
        print()
        print()
    torch.save(model, PATH_model)
    print("epochs:", epochs, "hidden_size:", hidden_size, "sent_len:", sent_len, "use_sent:", use_sent)


def evaluate(data, sent_len, file_name, device, output_path):
    print("load_model")
    model = torch.load(PATH_model)
    model.eval()
    with open('./model/id2word.pickle', 'rb') as file:
        id2word = pickle.load(file)
    print("start predict")
    with torch.no_grad():
        data = torch.Tensor(np.array(data))
        data_num = data.shape[0]
        dummy1 = torch.zeros(data_num, sent_len).to(device)
        dummy2 = torch.zeros(data_num, sent_len).to(device)
        data = data.to(device)
        _, pre_sent = model(data, dummy1, dummy2, device, 0)
        pre_sent = np.transpose(pre_sent)
        
        file = open(output_path + PATH_save, "w")
        for name, sent in zip(file_name, pre_sent):
            pre = turn_tar_to_sent(sent, id2word)
            if pre == "":
                pre = "A"
            answer = name + ',' + pre + '\n'
            file.write(answer)
        file.close()         
    print("end")


def turn_tar_to_sent(vec, id2word):
    temp = ""
    for i in range(len(vec)):
        if vec[i] == 2:
            temp = temp[:-1]
            temp = temp + '.'
            return temp
        elif vec[i] == 0 or vec[i] == 3 or vec[i] == 1:
            temp = temp + ''
        else:
            temp = temp + id2word[vec[i]] + ' '
    return temp
        
        
if __name__ == "__main__":
    arg = sys.argv
    if(len(arg) == 4):
        data_path = arg[1]
        output_path = arg[2]
        train_model = arg[3]
        train_model = int(train_model)
        main(data_path, output_path, train_model)
    else:
        print("input must be main.py <data_path> <output_path(.txt file)> <train_or_not>")
