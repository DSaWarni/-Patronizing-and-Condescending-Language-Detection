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


class Document2Vector():
    def __init__(self, df):      
      #df.columns = ['text','orig_label']
      self.df = df
      self.df.columns = ['text','orig_label']

      self.my_tags = list(df['orig_label'].unique())
      self.X_train, self.X_test, self.all_data, self.y_train, self.y_test = self.data_split()

      self.model_dbow = self.dbow_model_train()
      self.train_vectors_dbow, self.test_vectors_dbow = self.trained_vectors()
      
      self.accu_score = dict()
      self.classification_rep = dict()
      self.accu_score['Logistic Regression'] , self.classification_rep['Logistic Regression']  = self.logistic_regression()
      print('lr done')
      self.accu_score['Random Forest'] , self.classification_rep['Random Forest']  = self.rf()
      print('rf done')
      self.accu_score['Support Vector Machine'] , self.classification_rep['Support Vector Machine']  = self.SVM()
      print('svm done')
      self.accu_score['Support Vector Classifier'] , self.classification_rep['Support Vector Classifier']  = self.linear_SVC()
      print('svc done')
    #   self.accu_score['Multinomial Naive Bayes'] , self.classification_rep['Multinomial Naive Bayes']  = self.multinomial_NB()
    #   print('mnb done')


    def label_sentences(self, corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
        return labeled

    
    def data_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df.text, self.df.orig_label, random_state=0, test_size=0.3)
        X_train = self.label_sentences(X_train, 'Train')
        X_test = self.label_sentences(X_test, 'Test')
        all_data = X_train + X_test
        return X_train, X_test, all_data, y_train, y_test

    def dbow_model_train(self):
        model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=3, alpha=0.065, min_alpha=0.065)
        model_dbow.build_vocab([x for x in tqdm(self.all_data)])

        for epoch in range(30):
            model_dbow.train(utils.shuffle([x for x in tqdm(self.all_data)]), total_examples=len(self.all_data), epochs=1)
            model_dbow.alpha -= 0.002
            model_dbow.min_alpha = model_dbow.alpha

        return model_dbow
      
    def get_vectors(self, model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.dv[prefix]
        return vectors

    def trained_vectors(self):
        train_vectors_dbow = self.get_vectors(self.model_dbow, len(self.X_train), 300, 'Train')
        test_vectors_dbow = self.get_vectors(self.model_dbow, len(self.X_test), 300, 'Test')
        return train_vectors_dbow, test_vectors_dbow
    
    def logistic_regression(self):
        print('\n\n----------------------------Running Logistic Regression----------------------------\n\n')
        clf=LogisticRegression(class_weight='balanced') 
        # clf_parameters = {
        #         'solver':('newton-cg')#,'lbfgs'),
        #         } 
        # grid = GridSearchCV(clf, clf_parameters, n_jobs = -1, scoring='f1_macro',cv=10)
        logreg = clf.fit(self.train_vectors_dbow, self.y_train)
        y_pred = logreg.predict(self.test_vectors_dbow)
        
        # print("Best params: \n", grid.best_params_)
    
        accu_score = accuracy_score(y_pred, self.y_test)
        #print('accuracy %s' % accuracy_score(y_pred, y_test))
        classification_rep = classification_report(self.y_test,y_pred, target_names= self.my_tags)
        #print(classification_report(y_test, y_pred,target_names=my_tags))
        return accu_score, classification_rep
    
    def SVM(self):
        print('\n\n----------------------------Running SVM----------------------------\n\n')
        
        clf = svm.SVC(class_weight='balanced')  
        clf_parameters = {
                'kernel':('poly','linear','rbf','sigmoid'),
                'C':(0.1,0.5,1,2,10,50,100),
                }
        grid = GridSearchCV(clf, clf_parameters, n_jobs = -1, scoring='f1_macro',cv=10)
        svm_clf = grid.fit(self.train_vectors_dbow, self.y_train)
        y_pred = svm_clf.predict(self.test_vectors_dbow)
        
        print("Best params: \n", grid.best_params_)
    
        accu_score = accuracy_score(y_pred, self.y_test)
        #print('accuracy %s' % accuracy_score(y_pred, y_test))
        classification_rep = classification_report(self.y_test,y_pred, target_names= self.my_tags)
        #print(classification_report(y_test, y_pred,target_names=my_tags))
        return accu_score, classification_rep
        
    def linear_SVC(self):
        print('\n\n----------------------------Running Linear SVC----------------------------\n\n')
        
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
                'C':(0.1,1,2,10,50,100),
                }
        grid = GridSearchCV(clf, clf_parameters, n_jobs = -1, scoring='f1_macro',cv=10)
        linear_svc_clf = grid.fit(self.train_vectors_dbow, self.y_train)
        y_pred = linear_svc_clf.predict(self.test_vectors_dbow)
        
        print("Best params: \n", grid.best_params_)
    
        accu_score = accuracy_score(y_pred, self.y_test)
        #print('accuracy %s' % accuracy_score(y_pred, y_test))
        classification_rep = classification_report(self.y_test,y_pred, target_names= self.my_tags)
        #print(classification_report(y_test, y_pred,target_names=my_tags))
        return accu_score, classification_rep
    
    def rf(self):
        print('\n\n----------------------------Running Random Forest----------------------------\n\n')
        
        clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
        clf_parameters = {
                'criterion':('gini', 'entropy'), 
                'max_features':('sqrt', 'log2'),   
                'n_estimators':(30,50,100,200),
                'max_depth':(10,20),
                }
        grid = GridSearchCV(clf, clf_parameters, n_jobs = -1, scoring='f1_macro',cv=10)
        rf_clf = grid.fit(self.train_vectors_dbow, self.y_train)
        y_pred = rf_clf.predict(self.test_vectors_dbow)
        
        print("Best params: \n", grid.best_params_)
    
        accu_score = accuracy_score(y_pred, self.y_test)
        #print('accuracy %s' % accuracy_score(y_pred, y_test))
        classification_rep = classification_report(self.y_test,y_pred, target_names= self.my_tags)
        #print(classification_report(y_test, y_pred,target_names=my_tags))
        return accu_score, classification_rep

    def __repr__(self):
        return f'Accuracy scores:{self.accu_score} \n\n  Classification Report: \n{self.classification_rep}\n'
