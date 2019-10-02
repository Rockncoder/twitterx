import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['tweet', 'source', 'sentiment'])

from nltk.corpus import twitter_samples

for tweet in twitter_samples.strings('positive_tweets.json'):
    df.loc[len(df)] = [tweet, 'nltk.corpus.twitter_samples', 'positive']

for tweet in twitter_samples.strings('negative_tweets.json'):
    df.loc[len(df)] = [tweet, 'nltk.corpus.twitter_samples', 'negative']

airline_tweets = pd.read_csv('./Tweets.csv')
airline_df = airline_tweets[['text', 'airline_sentiment']]
airline_df = airline_df.rename(columns={'text': 'tweet', 'airline_sentiment': 'sentiment'})
airline_df['source'] = 'https://www.kaggle.com/crowdflower/twitter-airline-sentiment'

debate_tweets = pd.read_csv('./Sentiment.csv')
debate_df = debate_tweets[['text', 'sentiment']]
debate_df = debate_df.rename(columns={'text': 'tweet'})
debate_df['sentiment'] = debate_df['sentiment'].apply(lambda s: s.lower())
debate_df['source'] = 'https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment'

df = pd.concat([df, airline_df, debate_df], ignore_index=True)
print(df)

df[['tweet', 'sentiment']].groupby(['sentiment']).count().plot(kind='bar')
plt.show(block=True)
print('Total Tweets: ', len(df))
df.to_csv('twitter_sentiment_analysis.csv')

