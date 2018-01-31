import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from skimage.measure import compare_psnr
import pickle
import operator
import random
import utils
from model import create_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

def get_model(input_depth=32, output_depth=3, pad='zero'):
        net = create_model(downsample_channels = [128, 128, 128, 128, 128],
                   upsample_channels = [128, 128, 128, 128, 128],
                   skip_channels = [4, 4, 4, 4, 4],
                   input_channel_size = input_depth,
                   output_channel_size = img_np.shape[0],
                   upsample_mode='nearest',
                   need1x1=False,
                   padding_type=pad, 
                   activation_function=nn.LeakyReLU(0.2, inplace=True)
                ).type(dtype)

        return net

def get_net_input(img_size, input_depth=32, INPUT="noise"):
    return utils.get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)#utils.get_noise(input_depth, INPUT, (img_size[1], img_size[0])).type(dtype).detach()

def evaluate_model(net, net_input, img_np, img_mask_np, num_iter=6000,
                   show_every=500, report=True, figsize=10):
    pad = 'zero'
    OPTIMIZER = 'adam'

    INPUT = 'noise'
    input_depth = 32
    LR = 0.01 
    num_iter = 6001
    param_noise = False
    show_every = 500
    figsize = 10
    
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    #img_noisy_var = utils.np_to_var(img_noisy_np).type(dtype)
    # img
    psnr_history = []
    
    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_var = utils.np_to_var(img_np).type(dtype)
    mask_var = utils.np_to_var(img_mask_np).type(dtype)


    psnr_history = []
    def closure(i):
    
        #global i
    
        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n.data += n.data.clone().normal_()*n.data.std()/50
    
        out = net(net_input)
   
        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()
    
        psrn = compare_psnr(utils.var_to_np(out), img_np)
        psnr_history.append(psrn)
    
        if report:


            print ('Iteration %05d    Loss %f   PSNR %.3f' % (i, total_loss.data[0], psrn), '\r', end='')
            if  i % show_every == 0:
                out_np = utils.var_to_np(out)
                utils.plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
        
            #i += 1

        return total_loss
    
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    for j in range(num_iter):
        optimizer.zero_grad()
        closure(j)
        optimizer.step()
        
    if report:
        out_np = utils.var_to_np(net(net_input))
        q = utils.plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);
        
        data = {}
        data['psnr_history'] = psnr_history
        pickle.dump(data, open('inpainting_validation_psnr.p','wb'))

    max_index, max_value = max(enumerate(psnr_history), key=operator.itemgetter(1))
    return max_index, max_value

picture_paths = utils.get_picture_randomly(n=100)

data = {'max_psnrs': [], 'iteration': []}

i = 0
for path in picture_paths:
    
    # Make image and mask dividebly by a given number.
    dim_div_by = 64

    img_pil = utils.get_image(path)


    img_mask_pil_random = utils.get_text_mask(img_pil, str(random.randint(100000, 900000)))
    img_mask_pil = utils.crop_image(img_mask_pil_random, dim_div_by)
    img_pil      = utils.crop_image(img_pil, dim_div_by)

    img_np      = utils.pil_to_np(img_pil)
    img_mask_np = utils.pil_to_np(img_mask_pil)

    #img_np, img_mask_np = utils.get_original_and_corrupted_image(path)
    net = get_model(output_depth=img_np.shape[0])
    initial_input = get_net_input((img_np.shape[2], img_np.shape[1], img_np.shape[0]))
    max_iteration, max_psnr = evaluate_model(net, initial_input, img_np, img_mask_np, report=False)
    
    utils.save_statistics({path: [max_iteration, max_psnr]}, 'inpainting_validation_psnr_imageset_nohup.p')
    i += 1
    
    
    
