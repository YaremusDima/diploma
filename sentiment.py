import pandas as pd
import numpy as np
# import torch
import tqdm
import sys
from deeppavlov import build_model

model = build_model('rusentiment_convers_bert', download=True)

data_sentiment = pd.read_csv("sentiment.csv")
age = sys.argv[1]
gender = sys.argv[2]
print(f'run sentiment for age {age} and gender {gender}')

dataset_path = f'datasets/{age}_{gender}_0_150.csv'
try:
    df = pd.read_csv(dataset_path, delimiter='\t')
except:
    dataset_path = f'datasets/{age}_{gender}_0_150'
    df = pd.read_csv(dataset_path, delimiter='\t')

df = df if len(df) <= 1000000 else df.sample(1000000)
df.replace('þ<br />þ', '\n ', regex=True, inplace=True)
# df.head()
data = df['text'].tolist()

batch_size = 500
batched_sentiment = []
for i in tqdm.tqdm(range(0, len(data), batch_size)):
  batched_sentiment.append(model(data[i : i+batch_size]))

plus = 0
minus = 0
neutral = 0
speech = 0
for batch in batched_sentiment:
  for sent in batch:
    if sent == 'positive':
      plus += 1
    elif sent == 'negative':
      minus += 1
    elif sent == 'neutral':
      neutral += 1
    elif sent == 'speech':
      speech += 1

data_sentiment = pd.read_csv('sentiment.csv')
# data_sentiment

df_temp = pd.DataFrame({'age': [age], 'gender': [gender], 'positive': [plus], 'negative': [minus], 'neutral': [neutral], 'speech': [speech]})
# df_temp

data_sentiment = pd.concat((data_sentiment, df_temp))
# data_sentiment

data_sentiment.to_csv('sentiment.csv', index=False)