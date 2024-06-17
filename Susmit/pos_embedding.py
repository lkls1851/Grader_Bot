import torch
import torch.nn as nn
import random

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_length, num_emb_space, context_length):
        super(PositionalEmbedding, self).__init__()
        self.num_idx=vocab_length
        self.out_dim=num_emb_space
        self.context=context_length
        self.input_embedding=nn.Embedding(self.num_idx, self.out_dim)
        self.abs_embedding_pos=nn.Embedding(self.context, self.out_dim)

    def weights(self):
        return self.pos_embedding.weight
    
    def forward(self, x):
        torch.manual_seed(123)
        x=torch.tensor(x)
        ctx=len(x)
        return (self.input_embedding(x)+self.abs_embedding_pos(torch.arange(self.context)))
        # return (self.input_embedding(x)+self.abs_embedding_pos(torch.arange(ctx)))

