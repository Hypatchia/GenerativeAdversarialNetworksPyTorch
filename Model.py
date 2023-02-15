# Global Imports
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(42)


# Imports of functions and classes for this project
from noise import get_noise
from generator import Generator
from discriminator import Discriminator

# Set training parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

# Pick cpu as device
device = 'cpu'

    


# Inititialize 

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)




# define function for discriminator loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: generator model, returns an image given z-dimensional noise
        disc: discriminator model, returns a single-dimensional prediction of real/fake
        criterion: the loss function,
        real: a batch of real images
        num_images: the number of images the generator should produce, 
               also the length of the real images
        z_dim: dimension of the noise vector, a scalar
        device: device type
    Returns:
        disc_loss: 
    '''
 
    noise = get_noise(num_images , z_dim,device=device)
    fakes = gen(noise).detach()
    fake_prediction = disc(fakes)
    fake_loss = criterion(fake_prediction, torch.zeros_like(fake_prediction))
    real_prediction = disc(real)
    real_loss = criterion(real_prediction, torch.ones_like(real_prediction))
    disc_loss = (fake_loss + real_loss)/2
   
    return disc_loss



# define function for generator loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return 
    Parameters:
        gen: 
        disc: 
        criterion: 
        num_images: 
        z_dim: 
        device: 
    Returns:
        gen_loss:
    '''
    noise = get_noise(num_images , z_dim,device)
    fakes = gen(noise)
    fake_prediction = disc(fakes)
    gen_loss = criterion(fake_prediction, torch.ones_like(fake_prediction))
   
    return gen_loss