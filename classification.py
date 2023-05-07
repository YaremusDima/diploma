import pandas as pd
import numpy as np
import os
import tqdm
from pathlib import Path
import yaml
import argparse

from pymystem3 import Mystem
from razdel import tokenize, sentenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# load stopwords
script_dir=os.path.dirname(os.path.abspath(__file__))
with open(f'{script_dir}/ru_stopwords.yaml', 'r') as f:
    ru_stopwords = yaml.safe_load(f)
    ru_stopwords.append("_")

parser = argparse.ArgumentParser(description='Args to make classification')
parser.add_argument('age', type=int, help='age to classificate')
parser.add_argument('gender', type=int, help='gender to classificate, 1-female, 2-male')
parser.add_argument('data_dir', type=str, default='datasets_prepared', help='dir for datasets')
# parser.add_argument('result_dir', type=str, help='dir for result', default='result')
parser.add_argument('data_maxlen', type=int, help='max len for messages in dataset', default=500000)
args = parser.parse_args()

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc), deacc=True) if word not in ru_stopwords] for doc in texts]

def preprocess_fast(data, stem, step=10):
    i = 0
    texts_without_stopwords = [" ".join(txt) for txt in remove_stopwords(data)]
    lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    txtpart = lol(texts_without_stopwords, step)
    res = []
    for txtp in tqdm.tqdm(txtpart):
        alltexts = ' '.join([txt + ' br ' for txt in txtp])

        words = stem.lemmatize(alltexts)
        doc = []
        for txt in words:
            if txt != '\n' and txt.strip() != '':
                if txt == 'br':
                    res.append(doc)
                    doc = []
                else:
                    doc.append(txt)
    return res

def preprocess(stem, text):
    tokens = stem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in ru_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    # text = " ".join(tokens)
    
    return tokens

if __name__ == '__main__':
    dataset_path = f'{script_dir}\{args.data_dir}\{args.age}_{args.gender}_0_150_train.csv'
    dataset = pd.read_csv(dataset_path)
    # print(dataset.head())
    # dataset_short = dataset if len(dataset) <= args.data_maxlen else dataset.sample(args.data_maxlen)
    # dataset_short.replace('þ<br />þ', '\n ', regex=True, inplace=True)

    stem = Mystem()
    data = dataset['text'].tolist()
    preprocessed_fast_data = preprocess_fast(data, stem, 50000)

    # Create Dictionary
    id2word = corpora.Dictionary(preprocessed_fast_data)
    # Create Corpus
    texts = preprocessed_fast_data
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    for num_topics in range(1, 7):
        # Build LDA model
        print(f'build {num_topics} topics model')
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=2000,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
        for i in lda_model.print_topics():
            print(i)
        # with open(fr'{script_dir}\results\topics_{num_topics}_{args.data_maxlen}_{args.age}_{args.gender}.yaml', 'w') as file:
        #     yaml.dump(dict(lda_model.print_topics()), file)

        with open(fr'{script_dir}\results\topics_{num_topics}_{args.data_maxlen}_{args.age}_{args.gender}.txt', 'w', encoding='utf-8') as f:
            for key in dict(lda_model.print_topics(num_words=15)):
                f.write(str(key) + ' : ' + dict(lda_model.print_topics(num_words=15))[key] + '\n')

        if num_topics > 1:
            # Visualize the topics
            # pyLDAvis.enable_notebook()
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
            pyLDAvis.save_html(vis, fr'{script_dir}\results\topics_{num_topics}_{args.data_maxlen}_{args.age}_{args.gender}.html')