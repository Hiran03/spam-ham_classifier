# MM21B030

# all imports

import pandas as pd
import numpy as np
import os
import re
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
from sklearn.model_selection import KFold
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.utils import resample
from sklearn.base import clone

import pickle


def pre_processing(line):
    # 1. Strip extra spaces and tabs
    line = re.sub(r'\s+', ' ', line).strip()

    # 2. Strip special characters
    line = re.sub(r'[^\w\s]', '', line)  # Removes all characters except alphanumeric and whitespace

    # 3. Replace numbers with 'number'
    line = re.sub(r'\d+', 'number', line)

    # 4. Strip English stop words
    stop_words = set(stopwords.words('english'))
    line = ' '.join([word for word in line.split() if word.lower() not in stop_words])

    # 5. Replace links with 'link'
    line = re.sub(r'http\S+|www\S+|https\S+', 'link', line)

    # 6. Lowercase all characters
    line = line.lower()
    
    return line

class NaiveBayes():
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.p_spam = None
        self.p_spam_words = None
        self.p_ham_words = None
    
    def fit(self, X, y):
        # Preprocess the training data
        X_processed = X.apply(pre_processing)
        
        # Vectorize the training data
        X_vectorized = self.vectorizer.fit_transform(X_processed)
        
        # Separate spam and ham messages
        X_spam = X_vectorized[y == '1']
        X_ham = X_vectorized[y == '0']
        
        # Calculate the probability of spam (P(spam))
        self.p_spam = X_spam.shape[0] / (X_spam.shape[0] + X_ham.shape[0])
        
        # Calculate the probability of each word given spam (P(word|spam)) and ham (P(word|ham))
        self.p_spam_words = (X_spam.sum(axis=0) + 1) / (X_spam.sum() + X_vectorized.shape[1])
        self.p_ham_words = (X_ham.sum(axis=0) + 1) / (X_ham.sum() + X_vectorized.shape[1])
        
        return self
    
    def predict(self, X_test):
        
        X_processed = X_test.apply(pre_processing)
        X_vectorized = self.vectorizer.transform(X_processed)
        
        def prediction(x_vec):
            x_vec = x_vec.toarray()
            # Calculate log probabilities for numerical stability
            log_prob_spam = np.log(self.p_spam) + x_vec.dot(np.log(self.p_spam_words.T)) + (1 - x_vec).dot(np.log(1 - self.p_spam_words.T))
            log_prob_ham = np.log(1 - self.p_spam) + x_vec.dot(np.log(self.p_ham_words.T)) + (1 - x_vec).dot(np.log(1 - self.p_ham_words.T))
            
            # Return 1 if spam, 0 if ham
            return '1' if log_prob_spam > log_prob_ham else '0'
            
        
        return np.array([prediction(x) for x in X_vectorized])
    
    def get_params(self, deep=True):
        return {}
    
    def set_params(self, **params):
        return self

file_path = r"C:\Users\hiran\Desktop\SEM7\FML\assignment2\naivemodel.pkl"

# Open the file and load the model
with open(file_path, 'rb') as file:
    model = pickle.load(file)

data = []

for filename in os.listdir(dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(dir, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                data.append(content)
        except:
            print("Not Found")
            
data = pd.DataFrame(data)

predictions = model.predict(data)

predictions.replace({'1': 1, '0': 0})

print(predictions)

df = pd.DataFrame(predictions)
df.to_csv('output.csv', index=False)