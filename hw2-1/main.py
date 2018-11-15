import numpy as np
import torch
import torch.optim as optim

from load import load_everything, data_loader
from model import S2VTModel, LanguageModelCriterion


batch_size= 51
hidden_size = 600
embed_size = 256

epochs = 1000
lr = 0.001

encode_len = 15
decode_len = 10
used_sent = 10

teach_init_rate = 0.5

model = "S2S"

rnn_drop_rate = 0.5
encode_drop_rate = 0.5
decode_drop_rate = 0.3

PATH = model + '_' + str(rnn_drop_rate) + '_' + str(encode_drop_rate) + '_' + str(decode_drop_rate) + '_'
PATH_model = PATH + str(epochs) + '_' +str(hidden_size) + '_' + str(encode_len) + '_' + str(used_sent) + '.pt'
PATH_save = PATH + str(epochs) + '_' +str(hidden_size) + '_' + str(encode_len) + '_' + str(used_sent) + '.txt'


def main():
    # train_data_loader
    print("loading data")
    data, label, _ = load_everything("train")
    data = np.array(data)
    train_data_loader = data_loader(data, label, encode_len)
    
    # test_data_loader
    test_data, test_label, test_file_name = load_everything("test")
    test_data = np.array(test_data)
    test_data_loader = data_loader(test_data, test_label,
                                    encode_len, train_data_loader.word2id, 
                                    train_data_loader.id2word
                                    )
    # cuda
    print("enviroment setup")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = S2VTModel(
        train_data_loader.voc_size,
        encode_len,
        decode_len,
        hidden_size,
        embed_size,
        rnn_cell='lstm',
        encode_drop=encode_drop_rate, 
        decode_drop=decode_drop_rate
    )
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    train(model, opt, train_data_loader, test_data_loader, device, epochs, batch_size, teach_init_rate)
    evaluate(model, test_data_loader, test_file_name, device)
    

def train(model, opt, train_data_loader, test_data_loader, device, epochs, batch_size=100, teach_init_rate=0):
    model.train()
    crit = LanguageModelCriterion()
    teach_rate = np.linspace(0, epochs, epochs)
    teach_rate = np.exp(-(20 / epochs) * teach_rate)
    for i in range(epochs):
        print("epochs:", i)
        # to see the training sentence
        loss_ep, pre_sent_e, ans1, ans2 = 0, 0, 0, 0
        # start training
        for batch in range(train_data_loader.data_size//batch_size):
            # load data 
            data, target, mask = train_data_loader.load_on_batch(
                batch * batch_size, (batch + 1) * batch_size,
                i % used_sent)
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            
            # get prediction
            pre_prob, pre_sent = model(data, target, device, teach_init_rate * teach_rate[i])
            
            # get loss
            loss = crit(pre_prob, target, mask)
            
            # backward
            opt.zero_grad()
            loss.backward()
            
            # grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            
            # step
            opt.step()
            
            # showing things
            loss_ep += loss
            ans1 = target[0].cpu().numpy()
            ans2 = target[1].cpu().numpy()
            pre_sent_e = pre_sent
            
        # showing things
        av_loss = loss_ep/(train_data_loader.data_size//batch_size)
       
        train_sent = train_data_loader.turn_tar_to_sent(pre_sent_e[:, 0])
        train_sent2 = train_data_loader.turn_tar_to_sent(pre_sent_e[:, 1])
        train_ans1 = train_data_loader.turn_vec_to_sent(ans1)
        train_ans2 = train_data_loader.turn_vec_to_sent(ans2)
        print()
        print("train_sent:", train_sent)
        print("ans:", train_ans1)
        print("train_sent2:", train_sent2)
        print("ans2:", train_ans2)
        print()
        validation(model, test_data_loader, device, i, av_loss)
    torch.save(model, PATH_model)

        
def evaluate(model, test_data_loader, test_file_name, device):
    model.eval()
    crit = LanguageModelCriterion()
    
    # load data
    data, target, mask = test_data_loader.load_on_batch(0, 100, 0)
    data= data.to(device)
    dummy1 = torch.zeros(target.shape).to(device)
    
    # predict
    pre_prob, pre_sent = model(data, dummy1, device, 0)
    pre_sent = np.transpose(pre_sent)
    
    # save as file
    file = open("./MLDS_hw2_1_data/" + PATH_save + ".txt", "w")
    for name, sent in zip(test_file_name, pre_sent):
        pre = test_data_loader.turn_tar_to_sent(sent)
        if pre == "":
            pre = "A"
        answer = name + ',' + pre + '\n'
        file.write(answer)
    file.close() 
    
        
def validation(model, test_data_loader, device, i, av_loss):
    model.eval()
    crit = LanguageModelCriterion()
    # load data
    data, target, mask = test_data_loader.load_on_batch(
                0, 100, i % used_sent)
    data, target, mask = data.to(device), target.to(device), mask.to(device)
    dummy1 = torch.zeros(target.shape).to(device)
    
    # predict
    pre_prob, pre_sent = model(data, dummy1, device, 0)
    loss = crit(pre_prob, target, mask)
    # print
    ans1 = target[0].cpu().numpy()
    ans2 = target[1].cpu().numpy()
    test_ans1 = test_data_loader.turn_vec_to_sent(ans1)
    test_ans2 = test_data_loader.turn_vec_to_sent(ans2)
    test_sent1 = test_data_loader.turn_tar_to_sent(pre_sent[:, 0])
    test_sent2 = test_data_loader.turn_tar_to_sent(pre_sent[:, 1])
    print()
    print("test_sent:", test_sent1)
    print("ans:", test_ans1)
    print("test_sent2:", test_sent2)
    print("ans2:", test_ans2)
    print()
    print("test_loss:", loss)
    print("train loss:", av_loss)
    print()
    print()
    model.train() 
        
        
if __name__ == "__main__":
    main()