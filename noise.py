
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(42)








# define noise function
def get_noise(n_samples, z_dim, device ='cpu'):
    ''' 
    Function for generating random noise vector given sample size
    and dimension of noise vector 
    Parameters :
        n_samples : number of noise samples to generate
        z_dim : dim of noise vector , number of values in each sample 
        device : type of device : cuda, cpu ...
    
    '''

    return torch.randn(n_samples, z_dim, device=device)