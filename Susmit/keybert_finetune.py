from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from Susmit.preprocess import GetDataset

kw_model = KeyBERT()

path='Genre Classification Dataset/train_data.txt'
dataset=GetDataset(path_to_data=path)
df=dataset.fetch_data()
text=df['Description'][0]
docs=[text]

kw_model.extract_keywords(docs=docs, keyphrase_ngram_range=(1,1))
print(kw_model.extract_keywords(docs=docs, vectorizer=KeyphraseCountVectorizer()))


