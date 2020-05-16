__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Data Preprocessing
def preprocessing(text):
    # tokenize into words
    tokens = [word for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]

    print( "- tokenize into words -" )
    print( tokens )
    print()


print(preprocessing('i am the good man.here is'))