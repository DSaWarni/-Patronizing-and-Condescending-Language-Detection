import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.test.utils import common_texts
import multiprocessing

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder

import gensim
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import doc2vec
from gensim.models import Doc2Vec
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
import re
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

import csv,sys
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest,chi2 

#!pip3 install contractions
import contractions
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer


class data():
    
    def __init__(self, path):
        self.path = path
        self.df = self.data_read()
        #self.df_text, self.df_lemmatized, self.df_text_undersampled, self.df_lemmatized_undersampled = self.text_cleaner()
        
    def data_read(self):
        
        rows=[]

        with open(os.path.join(self.path)) as f:
            for line in f.readlines()[4:]:
                par_id=line.strip().split('\t')[0]
                art_id = line.strip().split('\t')[1]
                keyword=line.strip().split('\t')[2]
                country=line.strip().split('\t')[3]
                t=line.strip().split('\t')[4]
                l=line.strip().split('\t')[-1]
                
                if l=='0' or l=='1':
                    lbin=0
                    rows.append(
                        {'par_id':par_id,
                        'art_id':art_id,
                        'keyword':keyword,
                        'country':country,
                        'text':t, 
                        'label':lbin, 
                        'orig_label':l
                        }
                    )
                
                else:
                    lbin=1
                    rows.append(
                        {'par_id':par_id,
                        'art_id':art_id,
                        'keyword':keyword,
                        'country':country,
                        'text':t, 
                        'label':lbin, 
                        'orig_label':l
                        }
                    )
                
            df=pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label'])
        return df
    
    def text_cleaner(self):
        
        df = self.df
        # All words converted to lowercase
        df["text"] = df["text"].apply(lambda x: x.lower())

        # Remove Contractions
        df["text"] = df["text"].apply(lambda x: contractions.fix(x))

        # Non-ASCII from Text Corpus
        def remove_non_ascii(text):
            return re.sub(r'[^\x00-\x7f]',r'', text)
        df["text"] = df["text"].apply(lambda x: remove_non_ascii(x))

        # Removal of Special Characters from 
        def remove_special_characters(text):
            """
                Remove special special characters, including symbols, emojis, and other graphic characters
            """
            emoji_pattern = re.compile(
                '['
                u'\U0001F600-\U0001F64F'  # emoticons
                u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                u'\U0001F680-\U0001F6FF'  # transport & map symbols
                u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                u'\U00002702-\U000027B0'
                u'\U000024C2-\U0001F251'
                ']+',
                flags=re.UNICODE)
            return emoji_pattern.sub(r'', text)
        df["text"] = df["text"].apply(lambda x: remove_special_characters(x))

        # Remove Punctuations
        def remove_punct(text):
            """
                Remove the punctuation
            """
            return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)
        df["text"] = df["text"].apply(lambda x: remove_punct(x))

        # Tokenize the Text
        nltk.download('punkt')
        df['tokenized'] = df['text'].apply(word_tokenize)

        # Stopword Removal
        nltk.download("stopwords")
        stop = set(stopwords.words('english'))
        df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stop])

        # POS Tagging

        nltk.download("popular")
        nltk.download('brown')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        wordnet_map = {"N":wordnet.NOUN, 
                    "V":wordnet.VERB, 
                    "J":wordnet.ADJ, 
                    "R":wordnet.ADV
                    }

        train_sents = brown.tagged_sents(categories='news')
        t0 = nltk.DefaultTagger('NN')
        t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        t2 = nltk.BigramTagger(train_sents, backoff=t1)

        def pos_tag_wordnet(text, pos_tag_type="pos_tag"):
            """
                Create pos_tag with wordnet format
            """
            pos_tagged_text = t2.tag(text)
            
            # map the pos tagging output with wordnet output 
            pos_tagged_text = [(word, wordnet_map.get(pos_tag[0])) if pos_tag[0] in wordnet_map.keys() else (word, wordnet.NOUN) for (word, pos_tag) in pos_tagged_text ]
            return pos_tagged_text
        lst = []
        for i in range(len(df)):
            lst.append(pos_tag_wordnet(df['tokenized'][i]))
        df['pos_tagged'] = lst

        # Lemmatization
        def lemmatize_word(text):
            """
                Lemmatize the tokenized words
            """
            lemmatizer = WordNetLemmatizer()
            lemma = [lemmatizer.lemmatize(word, tag) for word, tag in text]
            return lemma

        #lemmatize_word(df['pos_tagged'][0])
        df['lemmatized'] = df['pos_tagged'].apply(lambda x: lemmatize_word(x))
        df['lemmatized_text'] = df['lemmatized'].apply(lambda x: " ".join(x))


        # INPUT DF 
        df_text = df[['text','orig_label']]
        df_lemmatized = df[['lemmatized_text','orig_label']]

        # UNDERSAMPLED
        y = df['orig_label']
        X = df[['text', 'lemmatized_text']]
        
        y = LabelEncoder().fit_transform(y)
        print(Counter(y))
        
        # define undersample strategy
        undersample = RandomUnderSampler()
        # fit and apply the transform
        X_under, y_under = undersample.fit_resample(X, y)
        # summarize class distribution
        print(Counter(y_under))

        df_text_undersampled = X_under.drop(columns = ['lemmatized_text'])
        df_text_undersampled['orig_labels'] = y_under
        df_text_undersampled = df_text_undersampled.astype(str)

        df_lemmatized_undersampled = X_under.drop(columns = ['text'])
        df_lemmatized_undersampled['orig_labels'] = y_under
        df_lemmatized_undersampled = df_lemmatized_undersampled.astype(str)
        
        return df_text, df_lemmatized, df_text_undersampled, df_lemmatized_undersampled

        
        
        
        

        


























