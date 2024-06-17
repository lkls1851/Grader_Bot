from yake_kwe import YAKE_kw


data_path='dataset/learning-agency-lab-automated-essay-scoring-2/train.csv'

yake=YAKE_kw(dataset_path=data_path)

df=yake.processed_data()

print(df.head())