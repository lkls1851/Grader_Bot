import torch
import torch.nn as nn
import random

class Embedding:
    def __init__(self, ids, num_emb_space):
        if type(ids)==int:
            self.tokens=[]
            self.tokens.append(ids)
        else:
            self.tokens=ids

        self.ids_tensor=torch.tensor(self.tokens)
        self.num_idx=max(self.ids_tensor)+1
        self.out_dim=num_emb_space
        self.embedding=nn.Embedding(self.num_idx, self.out_dim)

    def weights(self):
        return self.embedding.weight
    
    def forward(self, x):
        torch.manual_seed(123)
        x=torch.tensor(x)
        return self.embedding(x)

