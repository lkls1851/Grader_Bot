import yake
import pandas as pd
import numpy as np
from bpe_tokeniser import BPETokeniser
from pos_embedding import PositionalEmbedding
from attentionv1 import Attentionv1
from embedding import Embedding

language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 10

# custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, 
#                                             windowsSize=windowSize, top=numOfKeywords, features=None)

# dataset=pd.read_csv('dataset/learning-agency-lab-automated-essay-scoring-2/train.csv')
# tok=BPETokeniser()
# embed=PositionalEmbedding(vocab_length=50257, num_emb_space=5, context_length=10)
# def kwe_yake(text):
#     keywords=custom_kw_extractor.extract_keywords(text)
#     kw=''
#     for el in keywords:
#         kw+=el
#         kw+=' '
#     tokens=tok.encode(kw)
#     embed_vec=embed.forward()
#     return embed_vec

class YAKE_kw():
    def __init__ (self, dataset_path):
        self.data=dataset_path
        self.df=pd.read_csv(self.data)
        self.tok=BPETokeniser()
        # self.embed=PositionalEmbedding(vocab_length=50257, num_emb_space=5, context_length=10)
        self.custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, 
                                            windowsSize=windowSize, top=numOfKeywords, features=None)
    def kwe_yake(self, text):
        keywords=self.custom_kw_extractor.extract_keywords(text)
        kw=''
        for el in keywords:
            kw+=el[0]
            kw+=' '
        tokens=self.tok.encode(kw)
        embed=Embedding(ids=tokens, num_emb_space=5)

        embed_vec=embed.forward(tokens)
        return embed_vec
    
    def processed_data(self):
        embed_vec=[]
        for i in range(len(self.df)):
            txt=self.df['full_text'][i]
            vec=self.kwe_yake(text=txt)
            embed_vec.append(vec)

        self.df['Vectors']=embed_vec
        return self.df
