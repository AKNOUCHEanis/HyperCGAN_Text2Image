# -*- coding: utf-8 -*-

import numpy as np
import torch

def tensor_modulation_numpy(weight, styles):
    """ Performs a low-rank tensor modulation """
    
    weight=weight.detach().numpy()
    styles=styles.detach().numpy()
    
    c_out, c_in, kw, kh = weight.shape
    b = styles.shape[0]
    rank = styles.shape[1] // (c_out + c_in + kw + kh)
    styles = styles.reshape(b, rank, c_out + c_in + kh + kw )
    
    factor1 = np.expand_dims(styles[:,:, :c_out].reshape(b, rank, c_out), [-3, -2, -1])
    factor2 = np.expand_dims(styles[:,:, c_out: c_out+c_in].reshape(b, rank, c_in), [2, -2, -1])
    factor3 = np.expand_dims(styles[:,:, c_out+c_in: c_out+c_in+kh].reshape(b, rank, kh), [2, -3, -1])
    factor4 = np.expand_dims(styles[:,:, c_out+c_in+kh:].reshape(b, rank, kw), [2, -3, -2])
    
    modulating_tensor = factor1* factor2* factor3 * factor4
    modulating_tensor = modulating_tensor.sum(axis=1)/np.sqrt(rank)
    
    modulating_tensor = modulating_tensor/ np.sqrt(15)
    
    return torch.tensor(modulating_tensor)