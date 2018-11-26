import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage

from data_loader import Get_data_loader
from model import GAN

# model const
noise_size = 100
lr_d = 2e-4
lr_g = 2e-4
ngf = 128
ndf = 128
norm_method = "B_N"
gan_type="GAN"
clip=0.03
# train const
d_t = 5
g_t = 1
epochs = 200

# loader const
numWorkers = 4
imageSize = 64
batchSize = 64
dataPath = 'extra_data/'
savePath_img = './imgs/' + "dt_" + str(d_t) + '_'
savePath_model = './model/' + "dt_" + str(d_t) + '_'



# cuda
print("enviroment setup")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train():
    # model
    model = GAN(noise_size, lr_d, lr_g, ngf, ndf, norm_method=norm_method, gan_type=gan_type, clip=clip)
    model = model.to(device)
    
    # data
    data_loader = Get_data_loader(dataPath, imageSize, batchSize, numWorkers)
    
    # train
    for ep in range(epochs):
        for i, (imgs, _) in enumerate(data_loader):
            # set data
            imgs = imgs.to(device)
            noise = torch.randn(batchSize, noise_size).float().to(device)
            
            # train discriminator
            for _ in range(d_t):
                loss_d = model.train_d(imgs, noise, device)
                
            # train generator
            for _ in range(g_t):
                loss_g = model.train_g(noise, device)
                
            if i % 100 == 0:
                print("loss_d:", loss_d)
                print("loss_g:", loss_g)
                print()
                model.predict(savePath_img + "e_" + str(ep) + "_b_"+ str(i), imgs, device)
                
        if ep % 10 == 0:
            torch.save(model, savePath_model + str(ep) + '_' + norm_method + '_' +  gan_type + ".model")
            
train()