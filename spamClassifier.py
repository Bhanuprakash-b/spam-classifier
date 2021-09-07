# -*- coding: utf-8 -*-
#spam Classifier
import pandas as pd
messages=pd.read_csv('K:/nlp/spamclassifier/smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])
#data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps=PorterStemmer()
wordnet=WordNetLemmatizer()
corpus=[]
for i in range(len(messages)):
    review=re.sub('[^a-zA-A]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    #
#creating a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values
#training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.30,random_state=0)
#training with naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,Y_train)
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression().fit(X_train,Y_train)
Y1_p=logisticRegr.predict(X_test)
Y_pred=spam_detect_model.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(Y_test,Y1_p)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,Y1_p)