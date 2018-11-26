import torch
import torch.nn as nn
from generator import Generator_base, Generator_SN
from discriminator import Discriminator_base, Discriminator_SN

import numpy as np
import scipy.misc as misc

class GAN(nn.Module):
    def __init__(self, noise_size, lr_d, lr_g, ngf, ndf,
                 norm_method="B_N", gan_type="GAN", clip=0.01):
        super(GAN, self).__init__()
        # mode
        self.norm_method = norm_method
        self.gan_type = gan_type
        
        # const
        self.clip = clip
        self.noise_size = noise_size
        
        # module
        if norm_method == "B_N":
            self.g = Generator_base(noise_size, ngf)
            self.d = Discriminator_base(ndf)
        if norm_method == "S_N":
            self.g = Generator_SN(noise_size, ngf)
            self.d = Discriminator_SN(ndf)
        
        # opt
        if gan_type == "GAN":
            self.optG = torch.optim.Adam(self.g.parameters(), lr_g, betas=(0.5, 0.999)) 
            self.optD = torch.optim.Adam(self.d.parameters(), lr_d, betas=(0.5, 0.999)) 
        if gan_type == "WGAN":
            self.optG = torch.optim.RMSprop(params=self.g.parameters(), lr=lr_g)
            self.optD = torch.optim.RMSprop(params=self.d.parameters(), lr=lr_d)
        
        # loss
        self.BCE_loss = torch.nn.BCELoss()
        
    def train_d(self, imgs, noise, device):
        # predict true img
        pre_true, pre_true_sig = self.d(imgs)
        
        # predict fake img
        fake_imgs = self.g(noise).detach()
        pre_fake, pre_fake_sig = self.d(fake_imgs)
        
        # get loss
        if self.gan_type == "GAN":
            batch_size = noise.shape[0]
            true_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)
            loss = (self.BCE_loss(pre_fake_sig, fake_label) + self.BCE_loss(pre_true_sig, true_label)) / 2
        if self.gan_type == "WGAN":
            loss = -torch.mean(pre_true) + torch.mean(pre_fake)
        
        # gradient
        self.optD.zero_grad()
        loss.backward()
        if self.gan_type == "WGAN":
            torch.nn.utils.clip_grad_norm_(self.d.parameters(), self.clip)
        # step
        self.optD.step()
        
        return loss.cpu().data.numpy()

    def train_g(self, noise, device):
        # predict fake img
        fake_imgs = self.g(noise)
        pre_fake, pre_fake_sig = self.d(fake_imgs)
        
        # get loss
        if self.gan_type == "GAN":
            batch_size = noise.shape[0]
            true_label = torch.ones(batch_size, 1).to(device)
            loss = self.BCE_loss(pre_fake_sig, true_label)
        if self.gan_type == "WGAN":
            loss = -torch.mean(pre_fake)
        
        # gradient
        self.optG.zero_grad()
        loss.backward()
        if self.gan_type == "WGAN":
            torch.nn.utils.clip_grad_norm_(self.g.parameters(), self.clip)
        # step  
        self.optG.step()
        return loss.cpu().data.numpy()
        

    def predict(self, path, imgs, device):
        self.g.eval()
        ## for testing purpose
        img = imgs.cpu().numpy()[0]
        img = img.transpose(1, 2, 0) * 0.5 + 0.5
        misc.imsave("./imgs/test.jpg", img)
        
        # create noise
        noise = torch.randn((25, self.noise_size)).to(device)
        # create path
        path = path + "_" + self.norm_method + "_" + self.gan_type +".jpg"
        # start predict
        fake_imgs = self.g(noise)
        fake_imgs = fake_imgs.cpu().data.numpy()
        fake_imgs = fake_imgs.transpose(0, 2, 3, 1) * 0.5 + 0.5

        out_imgs = []
        for i in range(5):
            out_imgs.append([])
            for j in range(5):
                out_imgs[i] = np.concatenate(
                    (   
                        fake_imgs[i * 5 + 0], 
                        fake_imgs[i * 5 + 1], 
                        fake_imgs[i * 5 + 2],
                        fake_imgs[i * 5 + 3],
                        fake_imgs[i * 5 + 4]
                    ),
                    axis=1
                )
                out_imgs[i] = np.array(out_imgs[i])
        out_imgs = np.concatenate(
                    (   
                        out_imgs[0], 
                        out_imgs[1], 
                        out_imgs[2],
                        out_imgs[3],
                        out_imgs[4]
                    ),
                    axis=0
        ) 
        out_imgs = np.array(out_imgs)    
        misc.imsave(path, out_imgs)
