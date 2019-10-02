import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.tokenize.casual import TweetTokenizer

tweet_tokenizer = TweetTokenizer(strip_handles=True)
data = pd.read_csv('./twitter_sentiment_analysis.csv')
tweets = data['tweet'].values.astype(str)
sentiments = data['sentiment'].values.astype(str)

X_train, X_test, y_train, y_test = train_test_split(tweets, sentiments, test_size=0.2, shuffle=True)

vectorizer = CountVectorizer(lowercase=True, tokenizer=tweet_tokenizer.tokenize, ngram_range=(1,3))
vectorizer.fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)

classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                                class_weight=None, random_state=None, solver='lbfgs', max_iter=1000, multi_class='auto',
                                verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
classifier.fit(X_train_vectorized, y_train)
X_test_vectorized = vectorizer.transform(X_test)
score = classifier.score(X_test_vectorized, y_test)
print('Accuracy = ', score)
