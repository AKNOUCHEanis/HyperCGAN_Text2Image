# -*- coding: utf-8 -*-

from Model import HyperNetwork_TGS_Head, HyperNetwork_TGS_Body
import torch

if __name__=="__main__":
    
    dim_z = 256
    c_in =  512
    c_out = 512
    kh, kw = 3,3
    
    dim_latent1 = 256
    dim_latent2 = 300
    
    hypnet_body = HyperNetwork_TGS_Body(dim_z, dim_latent1, dim_latent2)

    hypnet_head = HyperNetwork_TGS_Head(dim_h = dim_latent2 , c_in=c_in, c_out=c_out, kh=kh , kw=kw)
    
    z = torch.rand(dim_z)
    
    h = hypnet_body.forward(z)
    w = hypnet_head.forward(h)
    print(w.shape)