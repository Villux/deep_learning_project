import os
import random
import pickle
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable

def pil_to_np(img_PIL):
    '''
    Converts image in PIL format to np.array.
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32)/255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1,2,0)

    return Image.fromarray(ar)

def np_to_tensor(img_np):
    '''
    Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    '''
    Converts image in numpy.array to torch.Variable.
    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(np_to_tensor(img_np)[None, :])

def var_to_np(img_var):
    '''
    Converts an image in torch.Variable format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.data.cpu().numpy()[0]

def get_image(path):
    img = Image.open(path)
    return img

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''
    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)
    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]
    return img.crop(bbox)

def fill_noise(data, noise_type='u'):
    if noise_type == 'u':
        data.uniform_()
    elif noise_type == 'n':
        data.normal_() 

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Variable of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for filling tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = Variable(torch.zeros(shape))
        
        fill_noise(net_input.data, noise_type)
        net_input.data *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_var(meshgrid)
    else:
        assert False
        
    return net_input


def get_image_grid(images_np, nrow=8):
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow=8, factor=1, interpolation=None):
    """
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1,2,0), interpolation=interpolation)
    plt.show()
    
    return grid

def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np

def get_picture_randomly(n=100, root_folder='data/image_set/101_ObjectCategories'):
    subfolders = [os.path.join(root_folder, folder) for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]
    images = []
    for folder in subfolders:
        images.extend([os.path.join(folder, image) for image in os.listdir(folder) if os.path.isfile(os.path.join(folder, image))])
    return random.sample(images, n)

def get_model_parameters(net):
    return sum([np.prod(list(p.size())) for p in net.parameters()]); 

def save_statistics(obj, filename):
    try:
        data = pickle.load(open(filename, "rb"))
    except FileNotFoundError:
        data = {}
    extend(data, obj)
    pickle.dump(data, open(filename,'wb'))
    
def get_original_and_corrupted_image(fname, sigma=25/255.):
    image = get_image(fname)
    img_pil = crop_image(image, d=32)
    img_np = pil_to_np(img_pil)
    _, img_noisy_np = get_noisy_image(img_np, sigma)
    return img_np, img_noisy_np

def extend(obj, items):
    for key, value in items.items():
        obj[key] = value
