# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import *
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from Models import Discriminator_StyleGAN2, CNN_Encoder, Text_Encoder
from Losses import discriminator_loss, DAMSM_loss

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    Batch_size =10
    Dim_embedding = 256
    
    fake_images = torch.rand((Batch_size, 3, Dim_embedding, Dim_embedding)).to(device)
    real_images = torch.ones((Batch_size, 3, Dim_embedding, Dim_embedding)).to(device)
    
    
    conditions = torch.rand((Batch_size, 1, Dim_embedding)).to(device)
    discriminator = Discriminator_StyleGAN2(Dim_embedding, rank=1).to(device)
    
    #test disciminator loss
    
    y_hat_fake = []
    for i in range(Batch_size):
        
        y_hat_fake.append(discriminator.forward(fake_images[i].unsqueeze(0), conditions[i]))
        
    y_hat_fake = torch.stack(y_hat_fake, dim=0)
    
    y_hat_real = []
    
    for i in range(Batch_size) :
        
        y_hat_real.append(discriminator.forward(real_images[i].unsqueeze(0), conditions[i]) )
        
    y_hat_real = torch.stack(y_hat_real, dim=0)
    
    print(y_hat_real)
    
    y_hat_wrong = []
    
    for i in range(1, Batch_size-1):
        
        y_hat_wrong.append(discriminator(real_images[i-1].unsqueeze(0), conditions[i+1]))
        
    y_hat_wrong = torch.stack(y_hat_wrong, dim=0)
        
    
    lossD = discriminator_loss(y_hat_real.squeeze(1), y_hat_fake.squeeze(1), y_hat_wrong.squeeze(1))
    
    
    #print(lossD)
    
    #DAMSM loss 1- SENTENCE
    
    #cnn encoder
    cnn_encoder = CNN_Encoder(Dim_embedding).to(device)
    sub_region_features_fake, image_embedding_fake = cnn_encoder(fake_images)
    
    #text encoder
    #text_encoder = Text_Encoder(1000, Dim_embedding, dim_hidden=100).to(device)
    #sentence_embed = text_encoder(conditions)
    sentence_embed = conditions.squeeze(1)
    
    labels_fake = torch.zeros(Batch_size,1)
    
    #print(image_embedding_fake.shape)
    #print(sentence_embed.shape)
    lossDAMSM = DAMSM_loss(image_embedding_fake, sentence_embed, labels_fake)
    
    
    lossD = lossD + lossDAMSM
    print(lossD)
    
    
    #test cnn_encoder
    """
    cnn_encoder = CNN_Encoder(dim_embedding).to(device)
    
    sub_region_features, image_embedding = cnn_encoder(img)
    
    print("sub_region_features :", sub_region_features.shape)
   
    print("Image embedding code :", image_embedding.shape)
    
    
    """
    #plt.imshow(img.permute(0,3,2,1)[0])
    #plt.show()
    
    #test Loss
    