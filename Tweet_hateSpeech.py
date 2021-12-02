import sys
import nltk
import sklearn
import pandas
import numpy

print('Python:{}'.format(sys.version))
print('NLTK:{}'.format(nltk.__version__))
print('Scikit-learn:{}'.format(sklearn.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Numpy:{}'.format(numpy.__version__))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#load the dataset of the sms messages
df = pd.read_csv(r"C:\Users\mourinho\Downloads\TwitterHate.csv")

#.Using regular expressions, remove user handles. These begin with '@’.
#.Using regular expressions, remove URLs.
#.Remove ‘#’ symbols from the tweet while retaining the term.
import re
cleaned = []
txt = list(df['tweet'])
for i  in txt:
    t = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",i).split())
    cleaned.append(re.sub(r'^RT[\s]+', '', t))
df['Cleaned_tweet'] = cleaned

#store the SMS message data
#this a series data object
tweeter =df['Cleaned_tweet']
print(tweeter[:10])

#.Remove stop words.
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

processed = tweeter.apply(lambda x: ' '.join(
     term for term in x.split() if term not in stop_words))

#Remove word stems using a porter stemmer
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(
      ps.stem(term) for term in x.split()))

import nltk
from nltk.tokenize import word_tokenize

# create bag-of-words
all_words = []
for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(10)))

# Join the tokens back to form strings. This will be required for the vectorizers.
from nltk.tokenize.treebank import TreebankWordDetokenizer
all_words = TreebankWordDetokenizer().detokenize(all_words)

x = processed
y = df['label']

#3.	Perform train_test_split using sklearn.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,  test_size = 0.25, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=5000, stop_words='english')
# TF-IDF feature matrix
tfidftrain = tfidf_vectorizer.fit(x_train,y_train)
tfidftest = tfidf_vectorizer.fit_transform(x_test,y_test)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1000)
x = cv.fit_transform(processed).toarray()
y = df['label']

print(x.shape)
print(y.shape)

#3.	Perform train_test_split using sklearn.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,  test_size = 0.25, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, f1_score

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

# Performance accurancy

print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print(f'Recall score: {recall_score(y_test,y_pred)}')
print("f1 score :", f1_score(y_test, y_pred))

Class_imbalance = df['label'].value_counts().sort_index()
Class_imbalance.plot.bar(figsize = (8, 6),linewidth = 2,edgecolor = 'k',title="Hate Labels")
Class_imbalance

df['label'].value_counts()/df.shape[0]
# define class weights
w = {0:1, 1:93}

# define model
lg = LogisticRegression(random_state=13, class_weight=w)
# fit it
lg.fit(x_train,y_train)
# test
y_pred = lg.predict(x_test)

# performance
print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print("f1 score :", f1_score(y_test, y_pred))
print(f'Recall score: {recall_score(y_test,y_pred)}')

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
# define hyperparameters
crange = np.arange(0.5, 20.0, 0.5)
w = [{0:1, 1:93}, {0:1, 1:100}]
hyperparam_grid = {"penalty": ["l1", "l2"]
                   ,"C": crange
                   ,"class_weight": w
                   ,"fit_intercept": [True, False]  }
kfolds = StratifiedKFold(2)

# logistic model classifier
lg = LogisticRegression(random_state=42)
# define evaluation procedure
grid = GridSearchCV(lg,hyperparam_grid,scoring="recall", cv=kfolds, n_jobs=-1, refit=True)
grid.fit(x_train, y_train)
print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

lg = LogisticRegression(random_state=42,C = 0.5, class_weight={0: 1, 1: 100}, fit_intercept= True, penalty = 'l2')
# fit it
lg.fit(x_train,y_train)
# test
y_pred = lg.predict(x_test)

# performance
print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print("f1 score :", f1_score(y_test, y_pred))
print(f'Recall score: {recall_score(y_test,y_pred)}')



