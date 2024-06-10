from keyphrase_vectorizers import KeyphraseCountVectorizer
from Susmit.preprocess import GetDataset

path='Genre Classification Dataset/train_data.txt'
dataset=GetDataset(path_to_data=path)
df=dataset.fetch_data()
text=df['Description'][0]
docs=[text]
vectorizer = KeyphraseCountVectorizer()

vectorizer.fit(docs)
keyphrases = vectorizer.get_feature_names_out()

print(keyphrases)
