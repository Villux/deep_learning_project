import numpy as np
import torch
import torch.nn as nn 

class Downsampler(nn.Module):
    '''
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    '''
    def __init__(self, n_planes, factor, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        super(Downsampler, self).__init__()
        
        assert phase in [0, 0.5], 'phase should be 0 or 0.5'
        
        support = 2
        kernel_width = 4 * factor + 1

        # note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(factor, phase, kernel_width, support=support, sigma=sigma)
        
        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch       

        self.downsampler_ = downsampler

        if preserve_size:

            if  self.kernel.shape[0] % 2 == 1: 
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)
                
            self.padding = nn.ReplicationPad2d(pad)
        
        self.preserve_size = preserve_size
        
    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x= input
        self.x = x
        return self.downsampler_(x)
        
def get_kernel(factor, phase, kernel_width, support=None, sigma=None):
    # factor  = float(factor)
    if phase == 0.5: 
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])
        
    assert support, 'support is not specified'
    center = (kernel_width + 1) / 2.

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):

            if phase == 0.5:
                di = abs(i + 0.5 - center) / factor  
                dj = abs(j + 0.5 - center) / factor 
            else:
                di = abs(i - center) / factor
                dj = abs(j - center) / factor

            pi_sq = np.pi * np.pi

            val = 1
            if di != 0:
                val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                val = val / (np.pi * np.pi * di * di)
                
            if dj != 0:
                val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                val = val / (np.pi * np.pi * dj * dj)
                
            kernel[i - 1][j - 1] = val
            
        
    kernel /= kernel.sum()
    
    return kernel

#a = Downsampler(n_planes=3, factor=2, kernel_type='lanczos2', phase='1', preserve_size=True)






#################
# Learnable downsampler

# KS = 32
# dow = nn.Sequential(nn.ReplicationPad2d(int((KS - factor) / 2.)), nn.Conv2d(1,1,KS,factor))
    
# class Apply(nn.Module):
#     def __init__(self, what, dim, *args):
#         super(Apply, self).__init__()
#         self.dim = dim
    
#         self.what = what

#     def forward(self, input):
#         inputs = []
#         for i in range(input.size(self.dim)):
#             inputs.append(self.what(input.narrow(self.dim, i, 1)))

#         return torch.cat(inputs, dim=self.dim)

#     def __len__(self):
#         return len(self._modules)
    
# downs = Apply(dow, 1)
# downs.type(dtype)(net_input.type(dtype)).size()
