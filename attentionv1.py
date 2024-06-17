import numpy as np
import torch
import torch.nn.functional as F
class Attentionv1():
    def __init__(self, inputs):
        self.input=inputs
    def attn_scores(self, query):
        res=[]
        for el in self.input:
            dot_val=torch.dot(query, el)
            np_val=dot_val.detach().numpy()
            np_val=np.float32(np_val)
            res.append(np_val)
        res=torch.tensor(res)
        return res
    def attn_weights(self,query):
        scores=self.attn_scores(query=query)
        sum=torch.sum(scores)
        weights=scores/sum
        return weights
    def attn_softmax(self, query):
        scores=self.attn_scores(query=query)
        weights=F.softmax(scores, dim=0)
        return weights
    def context_vec(self, query):
        weight=self.attn_softmax(query=query)
        con_vec=torch.zeros(query.shape)
        for i, xi in enumerate(self.input):
            w=weight[i]
            con_vec+=w*xi
        return con_vec
