# -*- coding: utf-8 -*-
"""Depression classificataion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hljyP0eeH-1pPh1pKl9z3DZBhJyXAAcy
"""

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/mental_health.csv')

df.head()

x = df['text']
y = df['label']

port_stem = PorterStemmer()

def stemming(content):
  pattern = '[^a-zA-Z]'
  replacement = ' '
  stemmed_content = re.sub(pattern , replacement , content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

x = x.astype(str)
nltk.download('stopwords')

x = x.apply(stemming)

x.head()

vectorizer = TfidfVectorizer()

vectorizer.fit(x)
X = vectorizer.transform(x)

print(X)

X_train ,X_test , Y_train , Y_test = train_test_split(X , y , test_size=0.20 , random_state=2)

model = LogisticRegression()

model.fit(X_train , Y_train)

X_test_predict = model.predict(X_test)

X_train_predict = model.predict(X_train)

log_test_accuracy = accuracy_score(X_test_predict , Y_test)
log_train_accuracy = accuracy_score(X_train_predict , Y_train)

print('This is train accuracy {} and test accuracy {}' .format(log_test_accuracy , log_train_accuracy))