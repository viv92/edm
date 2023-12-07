'''
### Program implementing a text-to-image model using EDM from Elucidating the Design Space of Diffusion Models Paper

## Features:
1. This is basically the Muse model, but we use a edm to predict the image latents
2. So the architecture is as follows:
2.1. The text (captions) are encoded using a pre-trained T5 encoder 
2.2. A pre-trained VQVAE is used to map between image pixel space and image latent space. An image is fed to VQVAE encoder to obtain the img latent
2.3. The edm is implemented by a T5 decoder (only this is trained, all other components are pre-trained and frozen)
2.4. The input to the T5 decoder is [time_emb, noised_img_latent, placeholder_for_denoised_img_latent]
2.5. The text embedding (obtained in 2.1 from T5 encoder) is fed as conditioning input to the T5 decoder via cross-attention
2.6. The denoised image latent (obtained from T5 decoder) are fed to VQVAE decoder to obtain the image

## Todos / Questions:
1. Classifier-free guidance
2. Does xattn based conditioning make sense or we need a purely causal decoder-only self attn based conditioning 
5. in dpct, note that time is diffusion time and not just positional int (as in case of LLM). So don't use sinusoidal embeddings for time
6. its important to have enough capacity in transformer backbone (d_model >= x_dim) 

'''

import os
import cv2
import math 
from copy import deepcopy 
from matplotlib import pyplot as plt 
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json 
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image, make_grid

# import T5 (we use the T5 Encoder only)
from transformers import T5Tokenizer, T5ForConditionalGeneration

# import VAE for loading the pretrained weights
from VAE_transformer import VAE_Transformer, init_transformer, patch_seq_to_img, img_to_patch_seq

from utils_dpct_gendim_crossattn import *


# prediction function that predicts denoised x using the diffusion net, with input-output conditioning
# x: noised input 
# t: diffusion time = added noise sigma
def prediction_function(net, x, t, class_label, sigma_data, t_eps):
    # sigma_t = t - t_eps 
    sigma_t = t 
    c_in = 1 / torch.sqrt( torch.pow(sigma_t,2) + sigma_data ** 2 )
    c_skip = (sigma_data ** 2) / ( torch.pow(sigma_t, 2) + (sigma_data ** 2) )
    c_out = (sigma_data * sigma_t) / torch.sqrt( (sigma_data ** 2) + torch.pow(t, 2) )
    c_noise = 0.25 * torch.log(sigma_t)
    # expand dims
    c_in = expand_dims_to_match(c_in, x) 
    c_skip = expand_dims_to_match(c_skip, x)
    c_out = expand_dims_to_match(c_out, x)
    # prop 
    x_conditioned = c_in * x 
    if use_cnoise:
        t = c_noise 
    out = net(x_conditioned, t, class_label)
    out_conditioned = c_skip * x + c_out * out 
    return out_conditioned

# function to map discrete step n to continuous time t
# NOTE that in EDM paper, authors follow reverse step indexing mapping [N-1, 0] to time interval [t_eps, T] = [sigma_min, sigma_max]
# Also NOTE that in EDM paper, this function is used only during sampling
def step_to_time(rho, t_eps, T, N, n):
    inv_rho = 1/rho 
    a = math.pow(t_eps, inv_rho)
    b = math.pow(T, inv_rho)
    return torch.pow( b + ((a-b) * n)/(N-1), rho) 

# function to calculate the list of all time steps for the given schedule
# NOTE that in EDM paper, step interval [0 ... N-1, N] corresponds to time interval [T ... t_eps, 0]
def calculate_ts(rho, t_eps, T, N):
    ts = [] 
    for n in range(0, N):
        t_n = step_to_time(rho, t_eps, T, N, torch.tensor(n))
        ts.append(t_n)
    # append t[N] = 0
    ts.append(torch.tensor(0.0))
    return torch.tensor(ts) 

# function to calculate loss weight factor (lambda) 
def calculate_lambda(sigma_t, sigma_data):
    return ( torch.pow(sigma_t,2) + sigma_data ** 2 ) / torch.pow( sigma_t * sigma_data , 2)

# function to expand dims of tensor x to match that of tensor y 
def expand_dims_to_match(x, y):
    while len(x.shape) < len(y.shape):
        x = x.unsqueeze(-1)
    return x 


# function to sample / generate img - deterministic sampling scheme using heun solver
# NOTE that step range [0 ... N-1, N] = time range [T ... t_eps, 0] = noise range [sigma_max ... sigma_min, 0]
def deterministic_sampling_heun_cfg(net, img_shape, rho, ts, class_label, sigma_data, N, cfg_scale):
    if class_label is not None:
        class_label = class_label.to(device)

    x_n = torch.randn(img_shape) * ts[0] # NOTE that initial noise x_0 ~ gaussian(0, T^2) and not x_0 ~ gaussian(0, identity)
    x_n = x_n.to(device)
    t_n = ts[0]
    t_n = t_n.unsqueeze(0).expand(img_shape[0]).to(device) # shape: [n_samples]
    t_n_xshape = expand_dims_to_match(t_n, x_n)

    for n in range(0, N): 

        t_n_plus1 = ts[n+1]
        t_n_plus1 = t_n_plus1.unsqueeze(0).expand(img_shape[0]).to(device) # expand to n_samples
        t_n_plus1_xshape = expand_dims_to_match(t_n_plus1, x_n)

        pred_x_cond = prediction_function(net, x_n, t_n, class_label, sigma_data, start_time)
        d_n_cond = (x_n - pred_x_cond) / t_n_xshape 
        pred_x_uncond = prediction_function(net, x_n, t_n, None, sigma_data, start_time)
        d_n_uncond = (x_n - pred_x_uncond) / t_n_xshape 
        # cfg on score = cfg on predicted noise d_n
        d_n = d_n_cond + cfg_scale * (d_n_cond - d_n_uncond)

        x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * d_n 

        if not (n == N-1):

            pred_x_plus1_cond = prediction_function(net, x_n_plus1, t_n_plus1, class_label, sigma_data, start_time)
            d_n_plus1_cond = (x_n_plus1 - pred_x_plus1_cond) / t_n_plus1_xshape  
            pred_x_plus1_uncond = prediction_function(net, x_n_plus1, t_n_plus1, None, sigma_data, start_time)
            d_n_plus1_uncond = (x_n_plus1 - pred_x_plus1_uncond) / t_n_plus1_xshape  
            # cfg on score = cfg on predicted noise d_n
            d_n_plus1 = d_n_plus1_cond + cfg_scale * (d_n_plus1_cond - d_n_plus1_uncond)

            x_n_plus1 = x_n + (t_n_plus1_xshape - t_n_xshape) * (d_n + d_n_plus1) * 0.5

        # for next iter 
        x_n = x_n_plus1 
        t_n = t_n_plus1 
        t_n_xshape = t_n_plus1_xshape

    return x_n 


# function to sample / generate img - stochastic sampling scheme using heun solver
# NOTE that step range [0 ... N-1, N] = time range [T ... t_eps, 0] = noise range [sigma_max ... sigma_min, 0]
def stochastic_sampling_heun_cfg(net, img_shape, rho, ts, class_label, sigma_data, N, cfg_scale):
    pass


# fetch dataset - using data loader
def cifar10_dl(img_size, batch_size):
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size, antialias=True),  # args.image_size + 1/4 *args.image_size
        # torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # equivalent to transforming pixel values from range [0,1] to [-1,1]
    ])
    dataset = CIFAR10(
        "./dataset_cifar",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    return dataloader


# utility function to freeze model
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False) 

# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer
        
# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)
        

# convert tensor to img
def to_img(x):
    x = 0.5 * x + 0.5 # transform img from range [-1, 1] -> [0, 1]
    x = x.clamp(0, 1) # clamp img to be strictly in [-1, 1]
    x = x.permute(0,2,3,1) # [b,c,h,w] -> [b,h,w,c]
    return x 

# function to save a generated img
def save_img_generated(x_g, save_path):
    gen_img = x_g.detach().cpu().numpy()
    gen_img = np.uint8( gen_img * 255 )
    # bgr to rgb 
    gen_img = gen_img[:, :, ::-1]
    cv2.imwrite(save_path, gen_img)
        


### main
if __name__ == '__main__':
    # hyperparams for vae (VAE_Transformer)
    latent_dim = 16
    img_size = 128 # 32 # resizing cifar imgs from 32x32 to 128x128 
    img_channels = 3
    img_shape = torch.tensor([img_channels, img_size, img_size])
    resize_shape = (img_size, img_size)
    img_latent_dim = latent_dim # as used in the pretrained VQVAE 

    patch_size = 16 # 4 # as required by the pretrained FSQ_Transformer
    assert img_size % patch_size == 0
    patch_dim = img_channels * (patch_size**2)
    seq_len = (img_size // patch_size) ** 2 # equal to num latents per item
    
    # hyperparams for VAE Transformer
    d_model_vae = patch_dim * 1
    n_heads_vae = 8
    assert d_model_vae % n_heads_vae == 0
    d_k_vae = d_model_vae // n_heads_vae 
    d_v_vae = d_k_vae 
    n_layers_vae = 6
    d_ff_vae = d_model_vae * 4
    dropout_vae = 0.1

    # hyperparams for custom decoder (DPCT)
    d_model_dpct = latent_dim * 16 
    n_heads_dpct = 2
    assert d_model_dpct % n_heads_dpct == 0
    d_k_dpct = d_model_dpct // n_heads_dpct 
    d_v_dpct = d_k_dpct 
    n_layers_dpct = 6
    d_ff_dpct = d_model_dpct * 4
    dropout_dpct = 0.

    # hyperparams for T5 (T5 decoder implements the consistency model backbone)
    d_model_t5 = 768 # d_model for T5 (required for image latents projection)
    max_seq_len_t5 = 512 # required to init T5 Tokenizer
    # dropout = 0. # TODO: check if we can set the dropout in T5 decoder

    # hyperparams for consistency training
    start_time = 0.002 # start time t_eps of the ODE - the time interval is [t_eps, T] (continuous) and corresponding step interval is [1, N] (discrete)
    end_time = 80 # 16 # end time T of the ODE (decreasing end time leads to lower loss with some improvement in sample quality)
    N_final =  35 # final value of N in the step schedule (denoted as s_1 in appendix C)
    rho = 7.0 # used to calculate mapping from discrete step interval [1, N] to continuous time interval [t_eps, T]
    sigma_data = 0.5 # used to calculate c_skip and c_out to ensure boundary condition
    P_mean = -1.2 # mean of the train time noise sampling distribution (log-normal)
    P_std = 1.2 # std of the train time noise sampling distribution (log-normal)
    use_cnoise = True  

    sampling_strategy = 'deterministic'
    n_samples = 16 

    num_train_steps_per_epoch = 118
    num_epochs = 850
    total_train_steps = num_train_steps_per_epoch * num_epochs
    train_steps_done = num_train_steps_per_epoch * 0
    lr = 1e-4 # 3e-4 
    batch_size = 512 # lower batch size allows for more training steps per diffusion process (but reduces compute efficiency)
    random_seed = 10
    sample_freq = int(total_train_steps / 60) 
    model_save_freq = int(total_train_steps / 5) 
    plot_freq = model_save_freq
    p_uncond = 0.1 # for cfg
    cfg_scale = 1.5
    resume_training_from_ckpt = False         

    hyperparam_dict = {}
    hyperparam_dict['t0'] = start_time
    hyperparam_dict['tN'] = end_time
    hyperparam_dict['N_final'] = N_final
    hyperparam_dict['sampleStrategy'] = sampling_strategy
    hyperparam_dict['lr'] = lr
    hyperparam_dict['batch'] = batch_size
    hyperparam_dict['dmodelDPCT'] = d_model_dpct
    hyperparam_dict['nheadsDPCT'] = n_heads_dpct
    hyperparam_dict['nlayersDPCT'] = n_layers_dpct
    hyperparam_dict['dropoutDPCT'] = dropout_dpct
    hyperparam_dict['pUncond'] = p_uncond
    hyperparam_dict['cfgScale'] = cfg_scale
    hyperparam_dict['useCnoise'] = use_cnoise 

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + ':' + str(v) 

    save_folder = './generated_muse_vae_transformer_edm_customDecoder_cfg_cifar' + hyperparam_str
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # vae ckpt path 
    vae_ckpt_path = '/home/vivswan/experiments/latent_diffusion/ckpts/VAE_transformer_cifar|Ldim:16|imgSize:128|patchSize:16|patchDim:768|seqLen:64|dModel:768|nHeads:8|dropout:0.1.pth' # path to pretrained vae 

    # t5 model (for encoding captions) 
    t5_model_name = 't5-base'
    # muse_vae_edm save ckpt path
    muse_ckpt_path = './ckpts/muse_vae_transformer_edm_customDecoder_cfg_cifar' + hyperparam_str + '.pt'

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load img dataset
    dataloader = cifar10_dl(img_size, batch_size)

    # load pretrained VAE in eval mode 
    # init transformer encoder
    encoder_transformer = init_transformer(patch_dim, seq_len, d_model_vae, d_k_vae, d_v_vae, n_heads_vae, n_layers_vae, d_ff_vae, dropout_vae, latent_dim * 2, device)
    # init transformer decoder
    decoder_transformer = init_transformer(latent_dim, seq_len, d_model_vae, d_k_vae, d_v_vae, n_heads_vae, n_layers_vae, d_ff_vae, dropout_vae, patch_dim, device)
    # init FSQ_Transformer 
    vae_model = VAE_Transformer(latent_dim, encoder_transformer, decoder_transformer, seq_len, device).to(device)
    vae_model = load_ckpt(vae_ckpt_path, vae_model, device=device, mode='eval')

    # init T5 tokenizer and transformer model
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length=max_seq_len_t5)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

    # delete t5_decoder to save ram 
    del t5_model.decoder 

    # init custom decoder (DPCT)
    max_seq_len_dpct = seq_len * 2 + 1 # [t, x_noised, x_denoised]
    condition_dim = d_model_t5 
    net = init_dpct(max_seq_len_dpct, seq_len, d_model_dpct, latent_dim, condition_dim, d_k_dpct, d_v_dpct, n_heads_dpct, n_layers_dpct, d_ff_dpct, dropout_dpct, device).to(device)

    # freeze vqvae, t5_encoder and ema_net
    freeze(vae_model)
    freeze(t5_model.encoder)

    # optimizer and loss criterion
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=lr)

    # load ckpt
    if resume_training_from_ckpt:
        net, optimizer = load_ckpt(muse_ckpt_path, net, optimizer, device=device, mode='train')

    # train

    train_step = train_steps_done
    epoch = 0
    ema_losses = []
    criterion = nn.MSELoss(reduction='none') # NOTE that reduction=None is necessary so that we can apply weighing factor lambda

    ts = calculate_ts(rho, start_time, end_time, N_final) # NOTE this is used only for sampling in the EDM approach

    pbar = tqdm(total=num_epochs)
    while epoch < num_epochs:

        # fetch minibatch
        pbar2 = tqdm(dataloader)
        for imgs, labels in pbar2:
        
            # tokenize labels
            cap_list = labels.tolist()
            cap_list = [str(x) for x in cap_list]
            cap_tokens_dict = t5_tokenizer(cap_list, return_tensors='pt', padding=True, truncation=True)

            imgs = imgs.to(device)
            cap_tokens_dict = cap_tokens_dict.to(device)

            with torch.no_grad():
                # convert img to sequence of patches
                x = img_to_patch_seq(imgs, patch_size, seq_len) # x.shape: [b, seq_len, patch_dim]

                mu, logvar = vae_model.encode(x)
                img_latents = vae_model.reparameterize(mu, logvar)

                # extract cap tokens and attn_mask from cap_tokens_dict
                cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                # feed cap_tokens to t5 encoder to get encoder output
                enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # enc_out.shape: [batch_size, cap_seqlen, d_model_t5]

            x = img_latents 
            y = enc_out 

            # for sampling 
            sample_caption_emb = y[:1] # NOTE that we sample one class label but generate n_sample imgs for that label
            sample_caption_emb = sample_caption_emb.expand(n_samples, -1, -1)

            # set labels = None with prob p_uncond
            if np.random.rand() < p_uncond: # TODO: explore the effect of no CFG versus CFG only during training versus CFG during training and sampling
                y = None


            # alternate way to sample time = sigma using change of variable (as used in EDM paper) 
            # NOTE that this directly gives the time t = sigma and not the step index n where t = ts[n]
            log_sigma = torch.randn(x.shape[0]) * P_std + P_mean 
            t_n = torch.exp(log_sigma).to(device)
            
            # get corresponding noised data points
            z = torch.randn_like(x)
            x_n = x + expand_dims_to_match(t_n, x) * z 
            # predict x_0
            pred_x = prediction_function(net, x_n, t_n, y, sigma_data, start_time) # pred_x.shape: [b, x_seq_len, patch_dim]
            
            # calculate loss 
            weight_factor = calculate_lambda(t_n, sigma_data).to(device)
            weight_factor = expand_dims_to_match(weight_factor, x)
            d = criterion(pred_x, x)
            loss = weight_factor * d 
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema_losses (for plotting)
            if ema_losses == []:
                prev_loss = loss.item()
            else:
                prev_loss = ema_losses[-1]
            curr_loss = prev_loss * 0.9 + loss.item() * 0.1
            ema_losses.append(curr_loss)

            train_step += 1
            pbar2.update(1)
            pbar2.set_description('loss:{:.10f}'.format(ema_losses[-1]))

            # save ckpt 
            if train_step % model_save_freq == 0:
                save_ckpt(device, muse_ckpt_path, net, optimizer)

            # sample
            if train_step % sample_freq == 0:
                
                net.eval()

                # sample points - equivalent to just evaluating the consistency function
                with torch.no_grad():

                    sample_caption_string = cap_list[0]

                    sample_shape = x[:n_samples].shape # since we want to sample 'n_sample' points

                    if sampling_strategy == 'deterministic':
                        sampled_img_latents = deterministic_sampling_heun_cfg(net, sample_shape, rho, ts, sample_caption_emb, sigma_data, N_final, cfg_scale)
                    else:
                        sampled_img_latents = stochastic_sampling_heun_cfg(net, sample_shape, rho, ts, sample_caption_emb, sigma_data, N_final, cfg_scale)

                    # decode img latents to pixels using vae decoder
                    gen_img_patch_seq = vae_model.decode(sampled_img_latents)

                    # convert patch sequence to img 
                    gen_imgs = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]

                    # gen_imgs = to_img(gen_imgs.data) #  [b,h,w,c]
                    # gen_img = gen_img.squeeze(0) # [h,w,c]
                    # # save img
                    # save_img_name = 'trainStep=' + str(train_step) + '_caption=' + sample_caption_string + '.png'
                    # save_path = save_folder + '/' + save_img_name
                    # save_img_generated(gen_img, save_path)

                    gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0,1)
                    grid = make_grid(gen_imgs, nrow=4)
                    save_image(grid, f"{save_folder}/trainStep={train_step}_caption={sample_caption_string}.png")

                net.train()

            if train_step % plot_freq == 0:
                # plot losses 
                plt.figure()
                l = int( len(ema_losses) / 2 ) # for clipping first half
                plt.plot(ema_losses[l:])
                plt.title('final_loss:{:.10f}'.format(ema_losses[-1]))
                plt.savefig(save_folder + f'/loss_trainStep={train_step}.png' )

        epoch += 1
        pbar.update(1)
        pbar2.close()


    pbar.close()
