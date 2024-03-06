import os
import torch
import torch.nn as nn
import torchvision
import  matplotlib.pyplot as plt
from tqdm import tqdm
from unet import Simple_Unet

class DDPM(nn.Module):
    def __init__(self, denoise_network, noise_steps, beta_start=0.0001, beta_end=0.02, device="cuda"):
        super().__init__()
        self.denoise_network = denoise_network
        self.noise_step = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        # betaの時間変化　linspace：等差数列を出力
        self.beta = torch.linspace(start=beta_start, end=beta_end, steps=noise_steps).to(device) 
        self.alpha = 1.0 - self.beta
        # torch.cumprod([1, 2, 3, 4], dim=0) -> tensor([1, 2, 6, 24])
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x, t): # x: image, t：noise_step
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])          # batch_size
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar[t])      # batch_size

        # a = torch.tensor([1, 2, 6, 24, 288]) -> a.shape = torch.Size([5])
        # at = a.reshape(-1, 1, 1, 1) -> at.shape = torch.Size([5, 1, 1, 1])
        # at = tensor([[[[ 1]]], [[[ 2]]], [[[ 6]]], [[[24]]], [[[ 5]]]])
        # at[3, 0, 0, 0] = tensor(24)
        sqrt_alpha_bar = sqrt_alpha_bar.reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.reshape(-1, 1, 1, 1)   
        # randn_like: inputと同じshapeのランダムテンソルを正規分布に従って生成
        # rand_like: 一様分布に従って生成
        epsilon = torch.randn_like(input=x)
        return sqrt_alpha_bar*x + sqrt_one_minus_alpha_bar*epsilon, epsilon
    
    def reverse_process(self, x, t):
        pred_noise = self.denoise_network(x, t)
        return pred_noise
    
    def sample(self, pred_noise, t, x):     # t=0のとき、x: pure noise(torch.randn(sample_size, channel=3, img_size_h, img_size_w).to(device))
        coef = 1.0 / torch.sqrt(1.0 - self.beta)
        coef_epsilon = self.beta / torch.sqrt(1.0 - self.alpha_bar)

        coef = coef.reshape(-1, 1, 1, 1) 
        coef_epsilon = coef_epsilon.reshape(-1, 1, 1, 1)

        pred_x = coef * (x - coef_epsilon*pred_noise)

        if t > 0:
            noise = torch.randn_like(input=pred_x)
            pred_x += torch.sqrt(self.beta)*noise
        
        return pred_x
