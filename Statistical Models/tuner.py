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


class tuner():
    def __init__(self,df):
        self.df = df
        self.df.columns = ['text','orig_label']
        self.labels = self.df['orig_label']
        self.trn_data, self.tst_data, self.trn_cat, self.tst_cat = self.data_split()
        self.result = self.evaluation_models()



    def data_split(self):
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(self.df.text, self.labels, test_size=0.20, random_state=42,stratify=self.labels)
        return trn_data, tst_data, trn_cat, tst_cat
    
    def evaluation_models(self):
        model_code = ['Naive_Bayes_Classifer', 'Linear_SVC', 'SVM', 'Logistic_Regression_Classifier', 'Random_Forest_Classifier']  

        for opt2 in model_code:  
            # Naive Bayes Classifer
            if opt2=='Naive_Bayes_Classifer':      
                clf=MultinomialNB(fit_prior=True, class_prior=None)  
                clf_parameters = {
                    'clf__alpha':(0,1),
                    }  
            # Linear SVC
            elif opt2=='Linear_SVC': 
                clf = svm.LinearSVC(class_weight='balanced')  
                clf_parameters = {
                    'clf__C':(0.1,1,2,10,50,100),
                    }
            # SVM  
            elif opt2=='SVM':
                clf = svm.SVC(class_weight='balanced')  
                clf_parameters = {
                    'clf__kernel':('poly','linear','rbf','sigmoid'),
                    'clf__C':(0.1,0.5,1,2,10,50,100),
                    }   
            # Logistic Regression Classifier    
            elif opt2=='Logistic_Regression_Classifier':
                clf=LogisticRegression(class_weight='balanced') 
                clf_parameters = {
                    'clf__solver':('newton-cg','lbfgs'),
                    }    

            # Random Forest Classifier    
            elif opt2=='Random_Forest_Classifier':
                clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
                clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20),
                    }     
            else:
                print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
                sys.exit(0)                 


            #print(' \n\n  ************************Applying --> '+opt2+'  ******************************* \n')
            
            pipeline, feature_parameters = self.feature_extractor(clf)

            print(' \n\n  ***** Applying --> ' +opt2+ '  ***** \n')

            predict = self.classifier(feature_parameters, clf_parameters, pipeline)

            print('\n ***** Printing Results for ---> ' +opt2+ ' ***** \n')
            self.Evaluation(predict)

            print('\n ***** APPLICATION OF ' +opt2+ ' FINISHED *****')
        return 'FINISHED'

    def feature_extractor(self, clf):
        # Feature Extraction

        pipeline = Pipeline([
            ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
            ('feature_selector', SelectKBest(chi2, k=1000)),         
            ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
            ('clf', clf),]) 
        # print('fLag 4')
        feature_parameters = {
            'vect__min_df': (2,3),
            'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
            'feature_selector__k': (100,500,1000)
            }
        return pipeline, feature_parameters

    def classifier(self, feature_parameters, clf_parameters, pipeline):

        # Classificaion
        parameters={**feature_parameters,**clf_parameters}
        # print('parameters are',parameters)
        # print('Flag 1')

        grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10, n_jobs=-1)          
        #print('Flag 2')
        
        grid.fit(self.trn_data,self.trn_cat)     
        #print('Flag 3')
        
        clf= grid.best_estimator_  
        print('***** Best Set of Parameters ***** \n\n')
        print(grid.best_params_)
        print(clf)

        predicted = clf.predict(self.tst_data)
        predicted =list(predicted)
        return predicted
              

    def Evaluation(self,predicted):
        # Evaluation
        print('\n Total documents in the training set: '+str(len(self.trn_data))+'\n')  
        print('\n Total documents in the test set: '+str(len(self.tst_data))+'\n')
        print ('\n Confusion Matrix \n')  
        print (confusion_matrix(self.tst_cat, predicted))  

        ac=accuracy_score(self.tst_cat, predicted) 
        print ('\n Accuracy:'+str(ac)) 
        
        pr=precision_score(self.tst_cat, predicted, average='weighted') 
        print ('\n Precision:'+str(pr)) 
    
        rl=recall_score(self.tst_cat, predicted, average='weighted') 
        print ('\n Recall:'+str(rl))
    
        fM=f1_score(self.tst_cat, predicted, average='micro') 
        print ('\n Micro Averaged F1-Score:'+str(fM))

        fm=f1_score(self.tst_cat, predicted, average='macro') 
        print ('\n Macro Averaged F1-Score:'+str(fm))
