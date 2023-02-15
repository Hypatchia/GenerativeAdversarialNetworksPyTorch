import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(42)





# define generator function
def generator(input_dim ,output_dim):

    '''
    Function to return a neural network representing the generator 
    Parameters :
        input_dim : dimension of input vector
        output_dim : dimension of output vector
    
    Returns :
        Generator Model having a set of layers.
    
    '''

    return nn.Sequential(
        # Linear Transformation Layer
        # y = xA' + b
        nn.Linear(input_dim,output_dim),
        # BatchNormalization
        nn.BatchNorm1d(output_dim),
        # Activation Layer
        nn.ReLU(inplace=True),

    )


# Build Generator Class 

class Generator(nn.Module) :
    """ 
    Generator Class
    Values :
    z_dim : dimension of noise vector 
    im_dim : dimension of images
            MNIST images are of dim 28 x 28 = 784
    hidden_im : inner dimension 
    

    """

    def __init__(self, z_dim = 10 ,im_dim = 784,hidden_dim = 128):
        super().__init__()

        self.gen = nn.Sequential(
            Generator(z_dim,hidden_dim),
            Generator(hidden_dim,hidden_dim * 2),
            Generator(hidden_dim * 2,hidden_dim * 4),
            Generator(hidden_dim * 4,hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        '''
        Function to complete a forward pass of the generator: 
        Given a noise tensor, returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
    
    