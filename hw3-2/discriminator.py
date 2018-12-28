import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as S_N


class Discriminator_BN_base(nn.Module):
    def __init__(self, 
                 ndf, 
                 hair_len,
                 eye_len,
                 d_leaky, sol=0.2,
        ):
        super(Discriminator_BN_base,self).__init__()
        # def activation
        if d_leaky == True:
            activation = nn.LeakyReLU(sol, True)
        else:
            activation = nn.ReLU(True)  
        # def module
        self.out = nn.Sequential(
            # [batch, 3, 64, 64] -> [batch, ndf, 32, 32]
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1),
            activation,
            # [batch, ndf, 32, 32] -> [batch, ndf * 2, 16, 16]
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(ndf * 2),
            activation,
            # [batch, ndf * 2, 16, 16] -> [batch, ndf * 4, 8, 8]
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(ndf * 4),
            activation,
            # [batch, ndf * 4, 8, 8] ->[batch, ndf * 8, 4, 4]
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            activation,
        )
        self.conv_h = nn.Conv2d(ndf * 8, 100, kernel_size=1)
        self.linear_h = nn.Linear(100 * 4 * 4, hair_len)
        self.conv_e = nn.Conv2d(ndf * 8, 100, kernel_size=1)
        self.linear_e = nn.Linear(100 * 4 * 4, eye_len)
        
        self.linear = nn.Linear((ndf * 8) * 4 * 4, 1)
#         self.linear_h = nn.Linear((ndf * 8) * 4 * 4, hair_len)
#         self.linear_e = nn.Linear((ndf * 8) * 4 * 4, eye_len)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax()
        
    def forward(self, imgs):
        batch_size = imgs.shape[0]
        x = self.out(imgs)
        
        h = self.conv_h(x)
        h = h.view(batch_size, -1)
        hair = self.linear_h(h)
        hair = self.soft(hair)
        
        e = self.conv_e(x)
        e = e.view(batch_size, -1)
        eye = self.linear_e(e)
        eye = self.soft(eye)
        
        x = x.view(batch_size, -1)
#         hair = self.linear_h(x)
#         hair = self.soft(hair)
#         eye = self.linear_e(x)
#         eye = self.soft(eye)
        x = self.linear(x)  # for WGAN
        x_sig = self.sig(x) # for GAN
        
        return x, x_sig, hair, eye
    

class Discriminator_BN_base2(nn.Module):
    def __init__(self, 
                 ndf, 
                 hair_len,
                 eye_len,
                 d_leaky, sol=0.2,
        ):
        super(Discriminator_BN_base2,self).__init__()
        # def activation
        if d_leaky == True:
            activation = nn.LeakyReLU(sol, True)
        else:
            activation = nn.ReLU(True)  
        # def module
        self.out = nn.Sequential(
            # [batch, 3, 64, 64] -> [batch, ndf, 32, 32]
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1),
            activation,
            # [batch, ndf, 32, 32] -> [batch, ndf * 2, 16, 16]
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(ndf * 2),
            activation,
            # [batch, ndf * 2, 16, 16] -> [batch, ndf * 4, 8, 8]
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(ndf * 4),
            activation,
            # [batch, ndf * 4, 8, 8] ->[batch, ndf * 8, 4, 4]
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            activation,
        )
        self.linear = nn.Linear((ndf * 8) * 4 * 4 , 100)
        self.linear2 = nn.Linear(100 + hair_len + eye_len, 1)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax()
        
    def forward(self, imgs, hair, eye):
        batch_size = imgs.shape[0]
        x = self.out(imgs)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = torch.cat((x, hair, eye), 1)
        x = self.linear2(x)
        x_sig = self.sig(x) 
        return x, x_sig, 0, 0