## We will implement byte pair encoding

## We will use tiktoken library developed by OpenAI
import tiktoken

class BPETokeniser:
    def __init__(self):
        self.tokeniser=tiktoken.get_encoding('gpt2')

    def encode(self, x):
        return self.tokeniser.encode(x)
    def decode(self, x):
        if type(x)==int:
            meta_list=[]
            meta_list.append(x)
            x=meta_list
        return self.tokeniser.decode(x)



