import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from Utils import tensor_modulation_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Text_Encoder(nn.Module):
    def __init__(self,vocabSize,dim_emb,dim_hidden,drop=0.5,pad=0):
        super().__init__()
        # Vocabulary Size
        self.vocabSize=vocabSize
        # Embedding dimension
        self.dim_emb=dim_emb
        # Hidden state dimension // 2 pour avoir hidden_state une fois 
        # les deux directions stackées
        self.dim_hidden=dim_hidden//2
        # module Embbeding 
        self.emb=nn.Embedding(vocabSize,dim_emb,padding_idx=pad).to(device)
        # module lstm 
        self.rnn=nn.LSTM(dim_emb, self.dim_hidden,dropout=drop,bidirectional=True).to(device)
        
        self.dropout = nn.Dropout(drop).to(device)
        
    def forward(self,x,x_len):
        """
        x : (Sequence,Batch)
        x_lens: Longueur de chaque séquence 
        h0 : hidden state initial
        
        return : 
            words_emb : (Sequence,)
        """
        # On calcule les embedding des token
        input_emb=self.dropout(self.emb(x))
        # On pack les séquences deja paddées 
        input_emb=pack_padded_sequence(input_emb,x_len,enforce_sorted=False).to(device)
        output_packed,(hn,_)=self.rnn(input_emb)
        # On dé-pack l'ouput et on le permute pour avoir batch*Sequence
        # words_emb : liste des embedding des mots
        # seq_emb : embedding de la séquence
        words_emb=pad_packed_sequence(output_packed)[0]
        seq_emb=hn.view(-1,2*self.dim_hidden).to(device)
        
        return words_emb.permute(1,0,2),seq_emb
    

class Discrete_StyleGAN2(nn.Module):
    def __init__(self,c_in,c_out,scale_factor=2):
        """
        c_in : nombre de channels en entré
        c_out : nombre de channels en sortie
        scale_factor: le facteur d'upsampling
        """
        super().__init__()
        self.c_in=c_in
        self.c_out=c_out
        self.scale_factor=scale_factor
        self.conv=nn.Conv2d(c_in, c_out, kernel_size=(3,3),stride=1).to(device)
        self.upsample=nn.Upsample(scale_factor=self.scale_factor).to(device)
        self.activation=nn.ReLU().to(device)
    
    def forward(self,x):
        f=self.conv(x)
        fsamp=self.upsample(f)
        fout=self.activation(fsamp)
        return fout
    
class INR_GAN (nn.Module):
    def __init__(self,dim_in,dim_out):
        """
        dim_in : dimension en entrée
        dim_out : dimension en sortie

        """
        super().__init__()
        self.dim_in=dim_in
        self.dim_out=dim_out

        self.linear=nn.Linear(dim_in,dim_out).to(device)
        self.activation=nn.ReLU().to(device)
    
    def forward(self,x):
        return self.activation(self.linear(x))
    
class GeneratorBlock(nn.Module):
    def __init__(self,c_in,c_out,scale_factor,dim_in,dim_out):
        """
        c_in : nombre de channels en entré
        c_out : nombre de channels en sortie
        scale_factor: le facteur d'upsampling
        """
        super().__init__()
        self.stylegan2=Discrete_StyleGAN2(c_in,c_out,scale_factor)
        #self.inr_gan=INR_GAN(dim_in, dim_out)
        
    def forward(self,x):
        return self.stylegan2(x)#,self.inr_gan(x)

class HyperNetwork_TGS_Body(nn.Module):
    """ HypeNet BODY with 2 MLP Layers """
    
    def __init__(self, dim_z, dim_latent1, dim_latent2):
        super().__init__()
        
        self.dim_z = dim_z
        self.dim_latent1 = dim_latent1
        self.dim_latent2 = dim_latent2
        
        self.linear1=nn.Linear(dim_z, dim_latent1).to(device)
        self.linear2=nn.Linear(dim_latent1,dim_latent2 ).to(device)
        
        
    def forward(self, z):
        
        h1 = self.linear1(z)
        
        h1=F.tanh(h1)
        
        h2 = self.linear2(h1)
        
        return F.tanh(h2)

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
        
        self.linear=nn.Linear(dim_h,rank*(c_in+c_out+kh+kw)).to(device)
        
    def forward(self,h):
        
        return self.linear(h)
    
class HyperNetwork_TDS_Body(nn.Module):
    """Hypnet Body for Discriminant conv2D with 1 mlp Layer"""
    
    def __init__(self, dim_embedding, c_in, c_out, kh, kw, rank):
        
        super().__init__()
        
        self.dim_embedding = dim_embedding
        self.c_in = c_in
        self.c_out = c_out
        self.kh = kh
        self.kw = kw
        self.rank = rank
        
        self.linear = nn.Linear(dim_embedding, rank*(c_in+c_out+kh+kw)).to(device)
        
    def forward(self, embedding):
        
        return self.linear(embedding)
    
class HyperNet_Block(nn.Module):
    """Hypnet block for convBlock """
        
    def __init__(self, dim_embedding ,c_in, c_out, rank):
        
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.rank = rank
        self.dim_embedding = dim_embedding
        
        self.hypnet1 = HyperNetwork_TDS_Body(self.dim_embedding, self.c_in, self.c_in, 3, 3, self.rank)
        self.hypnet2 = HyperNetwork_TDS_Body(self.dim_embedding, self.c_in, self.c_out, 3, 3, self.rank)
        self.hypnet3 = HyperNetwork_TDS_Body(self.dim_embedding, self.c_in, self.c_out, 1, 1, self.rank)
        
    def forward(self, h):
        return self.hypnet1(h), self.hypnet2(h), self.hypnet3(h)
    
    
class ConvBlock(nn.Module):
    """ Block de convolutions utilisé dans le descriminateur du GAN """
    
    def __init__(self, c_in, c_out):
        super().__init__()
        
        self.conv1 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.conv2 = nn.Conv2d(c_in, c_out, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(c_in, c_out, 1, stride=2)
        self.ReLU = nn.ReLU()
        
    def forward(self, input):
        
        out = self.ReLU(self.conv1(input))
        out = self.ReLU(self.conv2(out))
        
        reg = self.conv3(input)
        
        return (out + reg) /torch.sqrt(torch.tensor(2))



class Descriminator_StyleGAN2(nn.Module):
    """ Descriminateur pour classifier les images en Réelle ou Fake """
    def __init__(self, dim_embedding, rank):
        super().__init__()
        
        self.dim_embedding = dim_embedding
        self.rank = rank
        
        self.convInit = nn.Conv2d(3, 128, 3, padding=1)
        self.convs =nn.ModuleList([ConvBlock(128, 256),ConvBlock(256, 512)])
        self.convs.append(ConvBlock(512, 512))
        self.convFinal = nn.Conv2d(512,512,3, padding=1)
        
        self.hypnet = nn.ModuleList([HyperNetwork_TDS_Body(self.dim_embedding, 3, 128, 3,3, self.rank),
                                     HyperNet_Block(self.dim_embedding, 128, 256, self.rank),
                                     HyperNet_Block(self.dim_embedding, 256, 512, self.rank),
                                     HyperNet_Block(self.dim_embedding, 512, 512, self.rank),
                                     HyperNetwork_TDS_Body(self.dim_embedding, 512, 512, 3, 3, self.rank)])
        
        self.linears = nn.Sequential(nn.Linear(512*4*4, 512), nn.ReLU(), nn.Linear(512,1)) #Param to optimize
        
        
    def forward(self, input, c):
        """ input de shape : batch, channels, Hight, Width
            c : sentence information 
        """
        
        w0 = self.hypnet[0].forward(c)
        m0 = tensor_modulation_numpy(self.convInit.weight, w0)
        self.convInit.weight.data *= m0[0]
        
        out = self.convInit(input)
        out = nn.functional.relu(out)
        
        w11, w12, w13 = self.hypnet[1].forward(c)
        m11 = tensor_modulation_numpy(self.convs[0].conv1.weight, w11)
        self.convs[0].conv1.weight.data *= m11[0]
        m12 = tensor_modulation_numpy(self.convs[0].conv2.weight, w12)
        self.convs[0].conv2.weight.data *= m12[0]
        m13 = tensor_modulation_numpy(self.convs[0].conv3.weight, w13)
        self.convs[0].conv3.weight.data *= m13[0]
        
        out = self.convs[0].forward(out)
        
        w21, w22, w23 = self.hypnet[2].forward(c)
        m21 = tensor_modulation_numpy(self.convs[1].conv1.weight, w21)
        self.convs[1].conv1.weight.data *= m21[0]
        m22 = tensor_modulation_numpy(self.convs[1].conv2.weight, w22)
        self.convs[1].conv2.weight.data *= m22[0]
        m23 = tensor_modulation_numpy(self.convs[1].conv3.weight, w23)
        self.convs[1].conv3.weight.data *= m23[0]
    
        out = self.convs[1].forward(out)
        
        for i in range(4):
            
            w31, w32, w33 = self.hypnet[3].forward(c)
            m31 = tensor_modulation_numpy(self.convs[2].conv1.weight, w31)
            self.convs[2].conv1.weight.data *= m31[0]
            m32 = tensor_modulation_numpy(self.convs[2].conv2.weight, w32)
            self.convs[2].conv2.weight.data *= m32[0]
            m33 = tensor_modulation_numpy(self.convs[2].conv3.weight, w33)
            self.convs[2].conv3.weight.data *= m33[0]
        
            out = self.convs[2].forward(out)
           
        w4 = self.hypnet[4].forward(c)
        m4 = tensor_modulation_numpy(self.convFinal.weight, w4)
        self.convFinal.weight.data *= m4[0]
        
        
        out = self.convFinal(out)    
           
        out = out.view(-1, 512*4*4)
        out = self.linears(out)
        
        return out
    
 

    

        
    
    
    
    
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    