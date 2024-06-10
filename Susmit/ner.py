import pandas as pd 
import spacy 
import requests 
from bs4 import BeautifulSoup
nlp = spacy.load("en_core_web_sm")
pd.set_option("display.max_rows", 200)


content = "Congress leader from Ram Prasad has moved the Supreme Court against her  from the Lok Sabha over the cash-for-query allegations against ."

doc = nlp(content)

for ent in doc.ents:
	print(ent.text, ent.start_char, ent.end_char, ent.label_)



