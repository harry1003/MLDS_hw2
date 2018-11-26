import torch.utils.data as Data
import torchvision
from torchvision import transforms


def Get_data_loader(path, imageSize, batch_size, num_workers=0):
    tfs = transforms.Compose([
        transforms.Resize(imageSize),
        transforms.CenterCrop(imageSize), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    trainset = torchvision.datasets.ImageFolder(path ,transform=tfs)
 
    trainloader = Data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return trainloader
    