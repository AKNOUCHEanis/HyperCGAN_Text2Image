# -*- coding: utf-8 -*-

from Model import HyperNetwork_TGS_Head, HyperNetwork_TGS_Body
from Utils import tensor_modulation_numpy
import torch

if __name__=="__main__":
    
    dim_z = 256
    c_in =  512
    c_out = 512
    kh, kw = 3,3
    rank = 1
    
    dim_latent1 = 256
    dim_latent2 = 300
    
    hypnet_body = HyperNetwork_TGS_Body(dim_z, dim_latent1, dim_latent2)

    hypnet_head = HyperNetwork_TGS_Head(dim_h = dim_latent2 , c_in=c_in, c_out=c_out, kh=kh , kw=kw, rank=rank)
    
    z = torch.rand(dim_z)
    
    h = hypnet_body.forward(z)
    w = hypnet_head.forward(h)
    
    conv = torch.nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kh)
    weight = conv.weight.data
    
    M = tensor_modulation_numpy(weight, w)
    
    print(M.shape)