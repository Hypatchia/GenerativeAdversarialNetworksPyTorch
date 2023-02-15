# Global imports
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(42)



# define descriminator function

def discriminator(input_dim, output_dim):
    """ 
    Disciminator Function
    Function returns Discriminator Layers
    Params :
        input_dim : dimension of input vector
        output_dim : dimension of output vector

    Returns :
        Discriminator Neural Net
    
    """

    return nn.Sequential(
        nn.Linear(input_dim,output_dim),
        nn.LeakyReLU(0.2, inplace=True),

    )





# Build Discriminator Class
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            discriminator(im_dim, hidden_dim * 4),
            discriminator(hidden_dim * 4, hidden_dim * 2),
            discriminator(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim,1)
           
        )

    def forward(self, image):
        """ 
        Function to make a forward pass for discriminator
        Given fake images , return whether it's fake or real
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        
        """
        return self.disc(image)