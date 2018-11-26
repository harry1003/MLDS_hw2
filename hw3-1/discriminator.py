import torch.nn as nn
import torch.nn.utils.spectral_norm as S_N


class Discriminator_base(nn.Module):
    def __init__(self, ndf, sol=0.2):
        super(Discriminator_base,self).__init__()
        self.out = nn.Sequential(
            # [batch, 3, 64, 64] -> [batch, ndf, 32, 32]
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(sol, True),
 
            # [batch, ndf, 32, 32] -> [batch, ndf * 2, 16, 16]
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(sol, True),
 
            # [batch, ndf * 2, 16, 16] -> [batch, ndf * 4, 8, 8]
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(sol, True),
            
            # [batch, ndf * 4, 8, 8] ->[batch, ndf * 8, 4, 4]
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(sol, True),
        )
        self.linear = nn.Linear((ndf * 8) * 4 * 4, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, imgs):
        batch_size = imgs.shape[0]
        x = self.out(imgs)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x_sig = self.sig(x)
        return x, x_sig
    

class Discriminator_SN(nn.Module):
    def __init__(self, ndf, sol=0.2):
        super(Discriminator_SN,self).__init__()
        self.out = nn.Sequential(
            # [batch, 3, 64, 64] -> [batch, ndf, 32, 32]
            S_N(nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(sol, True),
 
            # [batch, ndf, 32, 32] -> [batch, ndf * 2, 16, 16]
            S_N(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)), 
            nn.LeakyReLU(sol, True),
            
            # [batch, ndf * 2, 16, 16] -> [batch, ndf * 4, 8, 8]
            S_N(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)), 
            nn.LeakyReLU(sol, True),
            
            # [batch, ndf * 4, 8, 8] ->[batch, ndf * 8, 4, 4]
            S_N(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(sol, True),
        )
        self.linear = nn.Linear((ndf * 8) * 4 * 4, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, imgs):
        batch_size = imgs.shape[0]
        x = self.out(imgs)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x_sig = self.sig(x)
        return x, x_sig