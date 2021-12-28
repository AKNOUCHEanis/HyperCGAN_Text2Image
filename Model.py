# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
    
"""*************** HyperNetwork Body and Head ****************"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HyperNetwork_TGS_Body(nn.Module):
    """ HypeNet BODY with 2 MLP Layers """
    
    def __init__(self, dim_z, dim_latent1, dim_latent2):
        super().__init__()
        
        self.dim_z = dim_z
        self.dim_latent1 = dim_latent1
        self.dim_latent2 = dim_latent2
        
        
        self.W1 = Parameter(torch.rand((self.dim_z, self.dim_latent1 )).to(device))
        self.B1 = Parameter(torch.rand((self.dim_latent1)).to(device))
        
        self.W2 = Parameter(torch.rand((self.dim_latent1, self.dim_latent2 )).to(device))
        self.B2 = Parameter(torch.rand((self.dim_latent2)).to(device))
        
        
    def forward(self, z):
        
        h1 = torch.matmul(z, self.W1) + self.B1
        
        h2 = torch.matmul(h1, self.W2) + self.B2
        
        return h2

class HyperNetwork_TGS_Head(nn.Module):
    """ Hypnet HEAD with 1 MLP Layer """
    
    def __init__(self, dim_h, c_in, c_out, kh, kw, rank):
        super().__init__()
        
        self.dim_h = dim_h
        self.c_in = c_in
        self.c_out = c_out
        self.kh = kh
        self.kw = kw
        self.rank = rank
        
        self.W = Parameter(torch.rand((self.dim_h, self.rank*(self.c_in + self.c_out + self.kh + self.kw))).to(device))
        self.B = Parameter(torch.rand((rank*(self.c_in + self.c_out + self.kh + self.kw))).to(device))
        
        
    def forward(self,h):
        
        res = torch.matmul(h, self.W) + self.B
        return res.view(1,self.rank*(self.c_in + self.c_out + self.kh + self.kw))
    


    
    
    
                

                
                
            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        
