import tweepy
from textblob import TextBlob

# Step 1 - Authenticate
consumer_key= '4UMrB2M4pyS3i439pITltkb4b'
consumer_secret= '1PhJzvMAmzI87s8VBfiSvn4MDU8c3mhbOvhqWe3zM8v6UWhr8i'

access_token='3085261123-LaRZtwjAEB6iNm443hqwocUZlN6g8wCteGIWmrx'
access_token_secret='V9ue1C7t7WBkIMHGvDJF5TBLUtsZi0UjfcmzqEt7gTmqi'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 3 - Retrieve Tweets
public_tweets = api.search('Trump')


#CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself

for tweet in public_tweets:
    print(tweet.text)
    
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)