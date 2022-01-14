# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def cosine_similarity(x1, x2, eps=1e-8):
    """Return the cosine similarity between x1 and x2 
    x1 and x2 have the same shape (batch, dim_embedding)
    """
    
    assert x1.shape[0] == x2.shape[0]
    assert x1.shape[1] == x2.shape[1]
    
    norm_x1 = torch.norm(x1, dim=1)  # batch x 1
    
    norm_x2 = torch.norm(x2, dim=1)  # batch x 1
    
    similarity = (x1 @ torch.t(x2)) / (norm_x1.reshape(-1,1)*norm_x2.reshape(-1,1)*torch.ones(x1.shape[0],x1.shape[0]))  # batch x batch
    
    return similarity.clamp(min=eps)
    
    

def words_loss(img_global_embed, sentence_embed, gamma1):
    """ sentence loss  between  images and sentences 
        image_global_embed : Batch x Dim_embedding
        sentence_embed : Batch x Dim_embedding
        return loss0, loss1
    """
    
    similarity_matrix = torch.mul(sentence_embed, torch.t(img_global_embed))
    
    similarity_matrix = torch.exp(similarity_matrix)
    
    sum_features = similarity_matrix.sum(dim=0) # 1 x 289
    
    similarity_matrix = similarity_matrix/sum_features
    
    
    
    
        
   

def sentence_loss(img_global_embed, sentence_embed, labels, gamma3=10, eps=1e-8 ):
    """ sentence loss  between  images and sentences 
        image_global_embed : Batch x Dim_embedding
        sentence_embed : Batch x Dim_embedding
        labels : Batch x 1
        gamma3 : hyperparametre
        
        return loss0, loss1
    """
    
    R_similarity = cosine_similarity(img_global_embed, sentence_embed, eps=eps)
    
    sum_images = R_similarity.sum(dim=0).reshape(-1,1).clamp(min=eps)
    
    sum_sentences = R_similarity.sum(dim=1).reshape(-1,1).clamp(min=eps)
   
    Proba_D_Q = torch.exp(gamma3*R_similarity)/torch.exp(gamma3*sum_sentences) #batch x batch
    Proba_D_Q = torch.diag(Proba_D_Q, 0).reshape(-1, 1) # batch x 1
   
    Proba_Q_D = torch.exp(gamma3*R_similarity)/torch.exp(gamma3*sum_images) 
    Proba_Q_D = torch.diag(Proba_Q_D, 0).reshape(-1, 1) # batch x 1
    
    loss_0 = nn.BCEWithLogitsLoss()(Proba_D_Q, labels)
    
    loss_1 = nn.BCEWithLogitsLoss()(Proba_Q_D, labels)
    
    return loss_0, loss_1
    
    
    
    
    
    
    
    
        
    



def DAMSM_loss(img_global_embed, sentence_embed, labels ,gamma3=5, eps=1e-8):
    
    
    loss_s_0, loss_s_1 = sentence_loss(img_global_embed, sentence_embed, labels, eps=eps)
    
    return loss_s_0 + loss_s_1



def discriminator_loss( y_hat_real, y_hat_fake, y_hat_wrong):
    """ y_hat_real : batch x 1
        y_hat_fake : batch x 1
        y_hat_wrong : (batch-1) x 1 Error in the descriptions
        
    """
    
    real_labels = torch.ones(y_hat_real.shape[0], 1)
    fake_labels = torch.zeros(y_hat_fake.shape[0], 1)
    
    
    real_errorD = nn.BCEWithLogitsLoss()(y_hat_real, real_labels)
    
    fake_errorD = nn.BCEWithLogitsLoss()(y_hat_fake, fake_labels)
    
    wrong_errorD = nn.BCEWithLogitsLoss()(y_hat_wrong, fake_labels[1:-1])
    
    
    
    return real_errorD +  fake_errorD + wrong_errorD
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    