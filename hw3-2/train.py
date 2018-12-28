import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
import csv
import numpy as np

from data_loader import Data_loader
from model import GAN

### GAN ###
# model const
noise_size = 100
lr_g = 2e-4
lr_d = 2e-4
ngf = 128
ndf = 128
g_leaky = True
d_leaky = True
norm_method = "B_N"
gan_type = "GAN"
g_leaky = True
clip = 0.03

# train const
d_t = 1
g_t = 3

epochs = 101
### GAN ###

# loader const
numWorkers = 4
imageSize = 64
batchSize = 64


dataPath = 'extra_data/'

# old version
savePath_img = './imgs/' + "d_t_" + str(d_t) + '_' + "g_t_" + str(g_t) + '_'
savePath_model = './model/' + "d_t_" + str(d_t) + '_' + "g_t_" + str(g_t) + '_'


# cuda
print("enviroment setup")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train():
    # data
    data_loader = Data_loader()
    # model
    model = GAN(
        noise_size, 
        lr_g, lr_d, ngf, ndf, 
        data_loader.hair_class,
        data_loader.eye_class,
        norm_method, gan_type,
        clip,
        g_leaky, d_leaky
    )
    model = model.to(device)
    
    # get test label
    t_hair_label = []
    t_eye_label = []
    with open("test_label.txt", newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            h = data_loader.hair_list.index(row[0])
            e = data_loader.eye_list.index(row[1])
            t_hair_label.append(h)
            t_eye_label.append(e)
    t_hair_label = np.array(t_hair_label)
    t_eye_label = np.array(t_eye_label)
    t_hair_label = data_loader.to_one_hot(t_hair_label, data_loader.hair_class)
    t_eye_label = data_loader.to_one_hot(t_eye_label, data_loader.eye_class)
    
    t_hair_label = t_hair_label.to(device)
    t_eye_label = t_eye_label.to(device)
            
    # train
    for ep in range(epochs):
        batch_num = data_loader.data_size // batchSize
        for i in range(batch_num):

            imgs, hair_label, eye_label = data_loader.load_on_batch(batchSize)
            # set data
            imgs = imgs.to(device)
            hair_label = hair_label.to(device)
            eye_label = eye_label.to(device)
            
            # old version
#             noise = torch.randn(batchSize, noise_size).float().to(device)
            noise = torch.rand(batchSize, noise_size).float().to(device)
            vector = torch.cat((noise, hair_label, eye_label), 1)
            
            # train discriminator
            for _ in range(d_t):
                loss_d = model.train_d(imgs, hair_label, eye_label, vector, device)  
            # train generator
            for _ in range(g_t):
                loss_g = model.train_g(hair_label, eye_label, vector, device)
                
#             if loss_g > loss_d:
#                 _ = model.train_g(hair_label, eye_label, vector, device)
                

            # print and visualize    
            if i % 100 == 0:
                print("loss_d:", loss_d)
                print("loss_g:", loss_g)
                print()
                model.predict(savePath_img + "e_" + str(ep) + "_b_"+ str(i), t_hair_label, t_eye_label, device)
        # save model       
        if ep % 10 == 0:
            torch.save(model, savePath_model + str(ep) + '_' + norm_method + '_' +  gan_type + ".model")

            
if __name__ == "__main__":
    train()