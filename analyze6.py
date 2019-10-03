# Grid Search
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.tokenize.casual import TweetTokenizer
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

tweet_tokenizer = TweetTokenizer(strip_handles=True)
data = pd.read_csv('./twitter_sentiment_analysis.csv')
tweets = data['tweet'].values.astype(str)
sentiments = data['sentiment'].values.astype(str)
tweets, sentiments = shuffle(tweets, sentiments)

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

classifier = GridSearchCV(pipeline, {
    'vectorizer__ngram_range': ((1,2), (2,3), (1,3)),
    'vectorizer__binary': (True, False),
}, n_jobs=-1, verbose=True, error_score=0.0, cv=5)
classifier.fit(tweets, sentiments)
print('Best Accuracy: ', classifier.best_score_)
print('Best Parameters: ', classifier.best_params_)

# this will take about 22 minutes to run
