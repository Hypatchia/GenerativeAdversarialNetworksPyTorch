
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(42)


# Imports for the specific project
from generator import Generator 
from generator import generator
from noise import get_noise

# define generator test Function 
def test_generator(in_dim,out_dim,test_size):
    """ 
    Function for testing the Generator 
    Parameters :
        in_dim : input vector dimension
        out_dim : output vector dimension
        test_size : number of obseravtions to test on

    Returns :
        set of assertions completed successfully
    """
    # Verify the generator block function
    #in_dim = 45
    #out_dim = 17
    #test_size =2000
    
    Gen = generator(in_dim,out_dim)
    assert len(Gen)==3
    assert type(Gen[0])==nn.Linear
    assert type(Gen[1])==nn.BatchNorm1d
    assert type(Gen[2])==nn.ReLU

    # Verify Input Output
    test_inpt = torch.randn(test_size,in_dim)
    test_outp = Gen(test_inpt)
    assert tuple(test_outp.shape)==(test_size, out_dim)
    assert test_outp.std() > 0.55
    assert test_outp.std() <0.65

    
# define test for noise function
def test_noise(n_samples,z_dim,device='cpu'):
    noise = get_noise(n_samples , z_dim,device)


    assert tuple(noise.shape) == (n_samples , z_dim)
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    assert str(noise.device).startswith(device)

    print('success')