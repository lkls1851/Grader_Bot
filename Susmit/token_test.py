## We will use Python's regular expression library 're'
import numpy as np
import pandas as pd
import re

f=open('verdict.txt', 'r')
text=f.read()
num_char=len(text)
# print(num_char,'\n')
# print(text[:99], '\n')

## Splitting by punctuations
result=re.split(r'([:,;.!?"()\']|--|\s)', text)
# print(result[:10])

## Removing whitespaces
result=[el.strip() for el in result if el.strip()]

# print(result[:99])

## Converting tokens to ID

unique_tokens=sorted(list(set(result)))
token_id={}
for i in range(len(unique_tokens)):
    token_id[unique_tokens[i]]=i

print(token_id)