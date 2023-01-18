'''
Positional Encoding reference : 
https://pytorch.org/tutorials/beginner/transformer_tutorial.html

'''
import torch
import torch.nn as nn
import math

class FFN(nn.Module):
    def __init__(self, emb_dim):
        super(FFN, self).__init__()
        self.emb_dim = emb_dim
        self.Linear1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.ReLU = nn.ReLU()
        self.Linear2 = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, inputs):
        return self.Linear2(self.ReLU(self.Linear1(inputs)))

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, device):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
    
    def pe(self, bat_size, seq_len):
        position = torch.arange(seq_len).unsqueeze(1)
        pe = torch.zeros(1,seq_len,self.emb_dim)
        div_term = torch.exp(torch.arange(0,self.emb_dim,2)*(-math.log(10000.0)/self.emb_dim))
        pe[0,:,0::2] = torch.sin(position * div_term)
        pe[0,:,1::2] = torch.cos(position*div_term)
        return torch.cat([pe]*bat_size,dim=0).to(self.device)
    
    def forward(self,x):
        bat_size, seq_len, _ = x.shape
        x = x + self.pe(bat_size, seq_len)
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, in_dim, emb_dim, device):
        super(EmbeddingLayer, self).__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.device = device
        self.Embedding = nn.Linear(self.in_dim, self.emb_dim, bias = False)
        self.PE = PositionalEncoding(self.emb_dim, self.device)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PE(x)
        return x

class TRKT(nn.Module):
    def __init__(self, in_dim, cfg):
        super(TRKT, self).__init__()
        self.query_dim = 200
        self.key_dim = in_dim
        self.emb_dim = cfg.edim
        self.out_dim = cfg.item
        self.device = cfg.device
        self.lam = cfg.lam
        self.query_embedding = nn.Linear(in_features = self.query_dim, out_features = self.emb_dim, bias = False)
        self.key_embedding = EmbeddingLayer(self.key_dim, self.emb_dim, self.device)
        self.Attention = nn.MultiheadAttention(self.emb_dim, num_heads = cfg.heads, dropout = cfg.dr, batch_first = True)
        self.FFN = FFN(self.emb_dim)
        self.classifier = nn.Linear(self.emb_dim, self.out_dim)
        self.LN1 = nn.LayerNorm(self.emb_dim)
        self.LN2 = nn.LayerNorm(self.emb_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = -1)

    def make_mask(self, x):
        return torch.triu(torch.ones(x,x), diagonal=1).type(torch.BoolTensor).to(self.device)

    def attn_weight(self, query, key, rel_mat):
        b, x = query.shape[0], query.shape[1]
        rel_mat = self.softmax(rel_mat)
        mask = self.make_mask(x)
        bat_mask = torch.stack([mask]*b,dim=0)
        td = torch.matmul(query, torch.transpose(key, -2, -1)) /math.sqrt(self.emb_dim)
        td = self.softmax(td)
        output = self.lam * td * bat_mask + (1-self.lam)*rel_mat * bat_mask 
        return output

    def get_rel_mat(self, query, text_emb):
        qnorm = torch.norm(query, dim = -1)
        tenorm = torch.norm(text_emb, dim = -1)
        rel_mat = torch.matmul(query, torch.transpose(text_emb,-2,-1))
        rel_mat = torch.div(rel_mat, qnorm.unsqueeze(dim = -1))
        rel_mat = torch.div(rel_mat, tenorm.unsqueeze(dim = -2))
        return rel_mat

    def forward(self, query, key, text_emb):
        attn_mask = self.make_mask(query.shape[1])
        rel_mat = self.get_rel_mat(query, text_emb)
        query = self.query_embedding(query)
        key = self.key_embedding(key)
        output = self.attn_weight(query, key, rel_mat)
        output = torch.matmul(output, key) + key
        output = self.LN1(output)
        output = self.FFN(output) + output
        output = self.LN2(output)
        output = self.classifier(output)
        del attn_mask
        return self.sigmoid(output)

    def posterior_predict(self, pred, target_item):
        target_item = torch.unsqueeze(target_item, -1) # Change a tensor into size (bat_size, len, 1) to match the dimension of pred
        posterior_item_predict = torch.gather(input = pred, index = target_item, dim = -1)
        posterior_item_predict = torch.squeeze(posterior_item_predict, dim = -1)
        return posterior_item_predict