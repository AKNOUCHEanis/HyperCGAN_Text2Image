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
    
def attention_mechanism( query, context, gamma1=5):
    """ calculation of context of a word 
        query : Batch x Sequence x Dim_Embedding
        context : Batch x 289 x Dim_Embedding
    """
    batch_size = query.shape[0]
    sequence = query.shape[1]
    
    query = torch.transpose(query, 1, 2) # Batch x Dim_Embedding x Sequence
    
    attention = torch.bmm(context, query) # Batch x 289 x Sequence
    
    attention = attention.view(batch_size*289, sequence)
    attention = nn.Softmax()(attention)
    
    attention = attention.view(batch_size, 289, sequence)
    attention = torch.transpose( attention, 1, 2).contiguous()
    attention = attention.view(batch_size*sequence, 289)
    
    attention = attention*gamma1
    attention = nn.Softmax()(attention)
    attention = attention.view(batch_size, sequence, 289)
    
    w = torch.bmm( torch.transpose(context, 1, 2), torch.transpose(attention, 1,2)) # batch x Dim_embedding x sequence
    
    return w, attention.transpose(1,2)
        
        
def words_loss(img_feature_embed, word_sentence_embed, labels, gamma1=5, gamma2=5, gamma3=10, eps=1e-8):
    """ sentence loss  between  images and sentences 
        image_feature_embed : Batch x 289 x Dim_embedding
        word_sentence_embed : Batch x SEQUENCE x Dim_embedding
        labels : Batch x 1
        return loss0, loss1
    """
    
   
    batch_size = word_sentence_embed.shape[0]
    sequence = word_sentence_embed.shape[1]
    dim_embedding = word_sentence_embed.shape[2]
    similarities = []
    
    for i in range(batch_size):
        
        words = word_sentence_embed[i,:,:].unsqueeze(0)
        
        words = words.view(1, sequence, dim_embedding ) # 1 x sequence x dim_embedding
        words = words.repeat(batch_size, 1, 1) # batch x sequence x dim_embedding
        
        context = img_feature_embed  # batch x 289 x dim_embedding
        
        w, attention = attention_mechanism(words, context, gamma1=5)
        
        # w : Batch x dim_embedding x sequence 
        # attention : batch x 289 x sequence
        
        #words = words.transpose(1, 2) # batch x sequence x dim_embedding
    
        w = torch.transpose(w, 1, 2).contiguous() # batch x sequence x dim_embedding
        
        
        # batch*sequence x dim_embeding
        words = words.view(batch_size*sequence, -1)
        
        w = w.view(batch_size*sequence, -1)
        
        similarity = torch.cosine_similarity(words, w, dim=1, eps=1e-8) # batch*sequence x 1
        
        similarity = similarity.view(batch_size, sequence) # batch x sequence
        
        similarity = torch.exp(gamma2 * similarity)
        similarity = similarity.sum(dim=1)
        similarity = torch.log(similarity) # batch x 1
        
        similarity = similarity.view(batch_size, 1)
        
        assert similarity.shape[0] == batch_size
        assert similarity.shape[1] == 1
        
        # similarities[i,j] : similarity between image i and description j 
        similarities.append(similarity)
        
    similarities = torch.cat(similarities,dim=1)
    
    print(similarities.shape)
    print(labels.shape)
        
    loss0 = nn.CrossEntropyLoss()(similarities, labels.view(batch_size).long())
    loss1 = nn.CrossEntropyLoss()(similarities.transpose(0,1), labels.view(batch_size).long())
        
    return loss0, loss1
        
   

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
    
    

def DAMSM_loss(img_feature_embed, img_global_embed, sentence_embed, word_sentence_embed, labels ,gamma1=5, gamma2=5, gamma3=10, eps=1e-8):
    
    
    loss_s_0, loss_s_1 = sentence_loss(img_global_embed, sentence_embed, labels, eps=eps)
    
    loss_w_0, loss_w_1 = words_loss(img_feature_embed, word_sentence_embed, labels, gamma1, gamma2, gamma3, eps=eps)
    
    return (loss_s_0 + loss_s_1)/2 + (loss_w_0 + loss_w_1 )/2



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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    