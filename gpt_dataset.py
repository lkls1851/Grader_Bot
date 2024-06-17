from bpe_tokeniser import BPETokeniser

tokeniser=BPETokeniser()

class GPTDataset:
    def __init__(self, path_to_text, context_size, stride):
        self.path=path_to_text
        self.tokeniser=tokeniser
        self.size=context_size
        self.stride=stride
        f=open(self.path, 'r')
        self.text=f.read()
        self.tokeniser=BPETokeniser()
        self.tokens=self.tokeniser.encode(self.text)
        self.data=[]
        for i in range(0, len(self.tokens)-self.size, self.stride):
            x=self.tokens[i:i+self.size]
            y=self.tokens[i+self.size]
            self.data.append([x,y])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x,y= self.data[index]
        return x, y

