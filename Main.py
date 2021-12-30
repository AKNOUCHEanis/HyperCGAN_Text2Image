# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import *
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from Models import Descriminator_StyleGAN2

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


"""
    
img=torch.rand((2,512,512))
desc=torch.randint(1000,(2,100))
inputImg=torch.randint(256,(2,1,128,128)).type(torch.float)


encoder=Text_Encoder(1000, 16, 32)
gen1=GeneratorBlock(1, 3, 2, 1, 1)
gen2=GeneratorBlock(3, 6, 2, 1, 1)

tgsbody=HyperNetwork_TGS_Body(132, 300, 400)
tgshead1=HyperNetwork_TGS_Head(400, 1, 3, 3, 3, 2)
tgshead2=HyperNetwork_TGS_Head(400, 3, 6, 3, 3, 2)

liste=torch.count_nonzero(desc.permute(1,0),dim=0).tolist() # pour le packedsequence
s=encoder(desc.permute(1,0),liste)[1]
noise=torch.rand((2,100))

desc.shape
s.shape
concat=torch.cat((desc,s),dim=1)
concat.shape

    
z=tgsbody(concat)
z1=tgshead1(z)
z2=tgshead2(z)


#Calcul des matrice de modulation M
m1=tensor_modulation_numpy(gen1.stylegan2.conv.weight,z1)
m2=tensor_modulation_numpy(gen2.stylegan2.conv.weight, z2)


for i in range (m1.shape[0]):
    #Maj des weights par modulation
    gen1.stylegan2.conv.weight.data*=m1[i]
    gen2.stylegan2.conv.weight.data*=m2[i]


f=gen1(inputImg)
f2=gen2(f)
f2=torch.mean(f2,dim=1)
f2[0]
plt.imshow(f2[0].detach().numpy())
plt.figure()
plt.imshow(inputImg[0].squeeze().detach().numpy())


import matplotlib.pyplot as plt    
import torch    
"""



if __name__=="__main__":
    
    img = torch.rand((1,3,256,256))
    
    descriminateur = Descriminator_StyleGAN2()
    
    out = descriminateur.forward(img)
    print(out)
    
    #plt.imshow(img.permute(0,3,2,1)[0])
    #plt.show()