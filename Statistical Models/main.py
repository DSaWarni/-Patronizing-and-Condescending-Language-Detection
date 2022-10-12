# Devesh Sharma 19103
# Praharsh Amit Nanavati 19222

from data_processing import data
from Document2Vector import Document2Vector
from tuner import tuner

#loading all datasets
path = 'dontpatronizeme_pcl.tsv'
df_text, df_lemmatized, df_text_undersampled, df_lemmatized_undersampled = data(path).text_cleaner()


#Applying Document2Vector embeddings and various models for DF_TEXT DATAFRAME
a = Document2Vector(df_text)
print('\n\n ***********Doc2Vec -- DF_TEXT***********')
##Printing results for Document2Vector for DF_TEXT DATAFRAME
print(a)

#Applying Document2Vector embeddings and various models for DF_TEXT_UNDERSAMPLED DATAFRAME
a = Document2Vector(df_text_undersampled)
print('\n\n ***********Doc2Vec -- DF_TEXT_UNDERSAMPLED***********')
##Printing results for Document2Vector for DF_TEXT_UNDERSAMPLED DATAFRAME
print(a)

#Applying Document2Vector embeddings and various models for DF_LEMMATIZED DATAFRAME
a = Document2Vector(df_lemmatized)
print('\n\n ***********Doc2Vec -- DF_LEMMATIZED***********')
##Printing results for Document2Vector for DF_LEMMATIZED DATAFRAME
print(a)

#Applying Document2Vector embeddings and various models for DF_LEMMATIZED_UNDERSAMPLED DATAFRAME
a = Document2Vector(df_lemmatized_undersampled)
print('\n\n ***********Doc2Vec -- DF_LEMMATIZED_UNDERSAMPLED***********')
##Printing results for Document2Vector for DF_LEMMATIZED_UNDERSAMPLED DATAFRAME
print(a)

del a

#Applying and printing results other embeddings and various models for DF_TEXT DATAFRAME
print("***** Printing for DF_TEXT *****")
a = tuner(df_text)

#Applying and printing results other embeddings and various models for DF_TEXT_UNDERSAMPLED DATAFRAME
print("***** Printing for DF_TEXT _UNDERSAMPLED*****")
a = tuner(df_text_undersampled)

#Applying and printing results other embeddings and various models for DF_LEMMATIZED DATAFRAME
print("***** Printing for DF_LEMMATIZED *****")
a = tuner(df_lemmatized)

#Applying and printing results other embeddings and various models for DF_LEMMATIZED_UNDERSAMPLED DATAFRAME
print("***** Printing for DF_LEMMATIZED_UNDERSAMPLED *****")
a = tuner(df_lemmatized_undersampled)

del a
        
        
