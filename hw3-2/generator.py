import torch.nn as nn
import torch.nn.utils.spectral_norm as S_N


class Generatorr_BN_base(nn.Module):
    def __init__(self, 
                 noise_size, 
                 ngf, 
                 g_leaky, sol=0.2
        ):
        super(Generatorr_BN_base, self).__init__()
        self.ngf = ngf
        # def activation
        if g_leaky == True:
            activation = nn.LeakyReLU(sol, True)
        else:
            activation = nn.ReLU(True)
        # def module
        self.linear = nn.Linear(noise_size, (ngf * 8) * 4 * 4)
        self.out = nn.Sequential(
            activation,
            # [batch, ngf * 8, 4, 4] -> [batch, ngf * 4, 8, 8]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(ngf * 4),
            activation,
            # [batch, ngf * 4, 8, 8] -> [batch, ngf * 2, 16, 16]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            activation,
            # [batch, ngf * 2, 16, 16] -> [batch, ngf * 1, 32, 32]
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            activation,
            # [batch, ngf * 1, 32, 32] -> [batch, 3, 64, 64]
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise):
        batch_size = noise.shape[0]
        x = self.linear(noise)
        x = x.view(batch_size, self.ngf * 8, 4, 4)
        x = self.out(x)
        return x
    

class Generatorr_SN_base(nn.Module):
    def __init__(self, 
                 noise_size, 
                 ngf, 
                 g_leaky, sol=0.2
        ):
        super(Generatorr_SN_base, self).__init__()
        self.ngf = ngf
        # def activation
        if g_leaky == True:
            activation = nn.LeakyReLU(sol, True)
        else:
            activation = nn.ReLU(True)
        # def module
        self.linear = nn.Linear(noise_size, (ngf * 8) * 4 * 4)
        self.out = nn.Sequential(
            activation,
            # [batch, ngf * 8, 4, 4] -> [batch, ngf * 4, 8, 8]
            S_N(nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1)),  
            nn.BatchNorm2d(ngf * 4),
            activation,
            # [batch, ngf * 4, 8, 8] -> [batch, ngf * 2, 16, 16]
            S_N(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ngf * 2),
            activation,
            # [batch, ngf * 2, 16, 16] -> [batch, ngf * 1, 32, 32]
            S_N(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ngf),
            activation,
            # [batch, ngf * 1, 32, 32] -> [batch, 3, 64, 64]
            S_N(nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1)),
            nn.Tanh()
        )
        
    def forward(self, noise):
        batch_size = noise.shape[0]
        x = self.linear(noise)
        x = x.view(batch_size, self.ngf * 8, 4, 4)
        x = self.out(x)
        return x

