import numpy as np
import re

class SimpleTokeniser():
    def __init__(self, path_to_text):
        self.path=path_to_text
        with open(self.path, 'r') as f:
            self.vocab=f.read()
        self.red_tokens=re.split(r'([:,;.!?"()\']|--|\s)', self.vocab)
        self.tokens=sorted(list(set([el.strip() for el in self.red_tokens if el.strip()])))
        self.hashmap={token:index for index,token in enumerate(self.tokens)}
        self.inverse_hash={index:token for index, token in enumerate(self.tokens)}

    def encode(self, x):

        tokens=re.split(r'([:,;.!?"()\']|--|\s)', x)
        tokens=[el.strip() for el in tokens if el.strip()]
        ids=[]
        for el in tokens:
        
            if el not in self.hashmap.keys():
                ids.append(-1)
        
            else:
                ids.append(self.hashmap[el])
        
        return ids
    
    def decode(self, id):

        string=''
        if type(id)==int:
            meta_list=[]
            meta_list.append(id)
            id=meta_list


        for el in id:

            if el not in self.inverse_hash.keys():
                pass
            elif el ==-1:
                pass 
            else:
                string+=(self.inverse_hash[el]+' ')

        return string

