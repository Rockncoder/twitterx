import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.tokenize.casual import TweetTokenizer
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

tweet_tokenizer = TweetTokenizer(strip_handles=True)
data = pd.read_csv('./twitter_sentiment_analysis.csv')
tweets = data['tweet'].values.astype(str)
sentiments = data['sentiment'].values.astype(str)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        lowercase=True,
        tokenizer=tweet_tokenizer.tokenize,
        ngram_range=(1,3))),
    ('classifier', LogisticRegression(
        solver='lbfgs',
        max_iter=400,
        multi_class='auto'))
])
tweets, sentiments = shuffle(tweets, sentiments)
print('Mean Accuracy= ', cross_val_score(pipeline, tweets, sentiments, cv=5).mean())
