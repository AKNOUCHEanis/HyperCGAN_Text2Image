import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from Utils import tensor_modulation_numpy

from torchvision import models
from torch.utils import model_zoo

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
    """Hypnet block for convBlock descriminator """
        
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


class FC_HyperNet(nn.Module):
    """ HyperNet for Linear Weights """
    def __init__(self, dim_embedding ,dim_in, dim_out, rank):
        
        super().__init__()
        
        self.dim_embedding = dim_embedding
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.rank = rank
        
        self.linear = nn.Linear(dim_embedding, rank*(dim_in+dim_out+1+1)).to(device)
        
    def forward(self, embedding):
        
        return self.linear(embedding)

class Discriminator_StyleGAN2(nn.Module):
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
                                     HyperNetwork_TDS_Body(self.dim_embedding, 512, 512, 3, 3, self.rank),
                                     FC_HyperNet(self.dim_embedding, 512*4*4, 512, self.rank),
                                     FC_HyperNet(self.dim_embedding, 512, 1, self.rank)])
        
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
        
        w5 = self.hypnet[5].forward(c)
        m5 = tensor_modulation_numpy( torch.ones((512, 512*4*4, 1, 1)),w5)
        self.linears[0].weight.data *= m5[0].view((512, 512*4*4))
        
        w6 = self.hypnet[6].forward(c)
        m6 = tensor_modulation_numpy( torch.ones((1, 512, 1, 1)), w6)
        self.linears[2].weight.data *= m6[0].view((1,512))
        
        out = self.linears(out)
        
        return out
    
 
class CNN_Encoder(nn.Module):
    
    def __init__(self, dim_embedding):
        super().__init__()
        
        self.dim_embedding =dim_embedding
        
        model = models.inception_v3().to(device)
        
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        
        state_dict = torch.load('Pretrained_DAMSM_Loss/image_encoder100.pth', map_location=torch.device("cpu"))

        
        for p in model.parameters():
            p.requires_grad = False
            
        
            
        #defining the model    
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        
        self.embed_features = nn.Conv2d(768, self.dim_embedding, kernel_size=1, stride=1, padding=0)
        self.embed_features.weight = nn.Parameter(state_dict.get("emb_features.weight"))
                
        self.embed_image = nn.Linear(2048, self.dim_embedding)
        self.embed_image.weight =  nn.Parameter(state_dict.get('emb_cnn_code.weight'))
        self.embed_image.bias =  nn.Parameter(state_dict.get('emb_cnn_code.bias'))
    
    def forward(self, x):
        """ x: shape (Batch, nchannels, hight, width )
            return : ( sub-region features (Batch, dim_embedding, 17, 17) , image embedding code (batch, dim_embedding) )
        """
        
        x = nn.Upsample(size=(299,299), mode='bilinear')(x)
        
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        # sub-region image features
        sub_region_features = x
        # N x 17 x 17 x 768

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        
        # N x 1 x 1 x 2048
        x = x.reshape(x.shape[0], -1)
        # N x 2048

        # global image features
        image_embedding = self.embed_image(x)
        # N x 512
        
        sub_region_features = self.embed_features(sub_region_features)
        
        return sub_region_features, image_embedding
        
    

        
    
    
    
    
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    