from bpe_tokeniser import BPETokeniser

class MyDataset:
    def __init__(self, path_to_text, context_size):
        self.path=path_to_text
        self.size=context_size
        f=open(self.path, 'r')
        self.text=f.read()
        self.tokeniser=BPETokeniser()
        self.tokens=self.tokeniser.encode(self.text)
        self.data=[]
        for i in range(len(self.tokens)-self.size):
            x=self.tokens[i:i+self.size]
            y=self.tokens[i+self.size]
            self.data.append([x,y])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x,y= self.data[index]
        w_x=self.tokeniser.decode(x)
        w_y=self.tokeniser.decode(y)
        return w_x, w_y