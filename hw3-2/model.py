import torch
import torch.nn as nn
from generator import Generatorr_BN_base, Generatorr_SN_base
from discriminator import Discriminator_BN_base, Discriminator_BN_base2

import numpy as np
import scipy.misc as misc

class GAN(nn.Module):
    def __init__(self, 
                noise_size,  
                lr_g, lr_d, ngf, ndf,
                hair_len,
                eye_len,
                norm_method="B_N", gan_type="GAN", 
                clip=0.01, 
                g_leaky=True, d_leaky=True
        ):
        super(GAN, self).__init__()
        # mode
        self.norm_method = norm_method
        self.gan_type = gan_type
        
        # const
        self.clip = clip
        self.noise_size = noise_size
        
        # module
        self.g = Generatorr_BN_base(noise_size + hair_len + eye_len, ngf, g_leaky)
        self.d = Discriminator_BN_base(ndf, hair_len, eye_len, d_leaky)

        # opt
        self.optG = torch.optim.Adam(self.g.parameters(), lr_g, betas=(0.5, 0.985)) 
        self.optD = torch.optim.Adam(self.d.parameters(), lr_d, betas=(0.5, 0.985)) 

        # loss
        self.BCE_loss = torch.nn.BCELoss()
        
    def train_d(self, imgs, hair_label, eye_label, noise, device):        
        # predict true img
        pre_true, pre_true_sig, true_hair, true_eye = self.d(imgs)
        # predict fake img
        fake_imgs = self.g(noise).detach()
        pre_fake, pre_fake_sig, fake_hair, fake_eye = self.d(fake_imgs)

        
        # get loss
        batch_size = noise.shape[0]
        true_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)
        
        img_loss = (self.BCE_loss(pre_fake_sig, fake_label) + self.BCE_loss(pre_true_sig, true_label)) / 2
        label_loss = self.BCE_loss(true_hair, hair_label) + self.BCE_loss(true_eye, eye_label)
        
        loss = img_loss + label_loss
        
        print_loss = loss.cpu().data.numpy()
        
        # gradient
        self.optD.zero_grad()
        loss.backward()

        # step
        self.optD.step()
        return print_loss

    def train_g(self, hair_label, eye_label, noise, device):
        # predict fake img
        fake_imgs = self.g(noise)
        pre_fake, pre_fake_sig, pre_hair_label, pre_eye_label = self.d(fake_imgs)
        
        # get loss
        batch_size = noise.shape[0]
        true_label = torch.ones(batch_size, 1).to(device)
        
        img_loss = self.BCE_loss(pre_fake_sig, true_label)
        label_loss = self.BCE_loss(pre_hair_label, hair_label) + self.BCE_loss(pre_eye_label, eye_label)
        
        loss = img_loss + label_loss
        
        print_loss = loss.cpu().data.numpy()

        # gradient
        self.optG.zero_grad()
        loss.backward()

        # step  
        self.optG.step()
        return print_loss
        
    def predict(self, path, hair, eye, device):
        self.g.eval()

        # create noise
        noise = torch.randn((25, self.noise_size)).to(device)
        vector = torch.cat((noise, hair, eye), 1)
        # create path
        path = path + "_" + self.norm_method + "_" + self.gan_type +".jpg"
        # start predict
        fake_imgs = self.g(vector)
        fake_imgs = fake_imgs.cpu().data.numpy()
        fake_imgs = (fake_imgs.transpose(0, 2, 3, 1) + 1) * 128
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
