"""
collect.py
"""
import sys
import time
import csv
import unicodecsv as csv
from TwitterAPI import TwitterAPI
import pickle
import numpy as np
from pathlib import Path

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    consumer_key = 'BtO1b9bcP4P9Ra1uYJXqBdix9'
    consumer_secret = 'N5DXG4AiICHt5h1gfMuGciJT21yFlEvPfbFivTJ07etgtU8BFM'
    access_token = '43856450-pHMoGhdowJBhe2yk5afGxAk2TkkNrgPYf64tF49mP'
    access_token_secret = 'EVVqqs3s6LQV8G6OVnJcEwgUAUlzwsaNNfMv2dsRnFtMH'
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def get_tweets(twitter,hashtags,tweets,since_id):

    while True:
        try:

            request = twitter.request('search/tweets',{'q':hashtags,'count':100,'lang':'en','max_id':since_id})
            for response in request:
                tweets.append(response)

            #print (len(tweets))
            return tweets
        except:
            print("Unexpected error:", sys.exc_info()[0])
            break

def get_friends(screen_name):
    return sorted([friend_id for friend_id in twitter.request('friends/ids',{'screen_name' : screen_name, 'count' :5000})])


def select_required(tweet):
    data = []

    for each_tweet in tweet:
        tweet_data = {}
        tweet_data['screen_name']=each_tweet['user']['screen_name']
        tweet_data['userid']= each_tweet['user']['id']
        tweet_data['description']=each_tweet['user']['description']
        tweet_data['tweet']=each_tweet['text']
        tweet_data['username']=each_tweet['user']['name']
        tweet_data['since_id']=each_tweet['id']
        data.append(tweet_data)
    return data


def open_file(tweets,filename):
    fname= filename+'.pkl'
    p = Path(fname)
    if p.is_file():
        with open(fname, "rb") as file:
            try:
                tweets = pickle.load(file)
            except EOFError:
                return tweets
    return tweets

def write_file(tweets,filename):
    fname=filename+'.pkl'
    print("Total Tweets collected for %s:" %filename,len(tweets))
    with open('collect.txt', 'a') as f:
        print("Total Tweets collected for %s:" %filename,len(tweets),file=f)

    pickle.dump(tweets, open(fname, 'wb'))


def main():
    twitter = get_twitter()
    text_file = open("collect.txt" , "w")
    print('Established Twitter connection.')
    print('Started collecting tweets From Twitter Based on hashtags')
    with open('collect.txt', 'a') as f:
        print('Established Twitter connection.',file=f)
        print('Started collecting tweets From Twitter Based on hashtags',file=f)
    hashtags=['#Celtics','#DetroitBasketball','#DefendTheLand','#WeTheNorth','#MADEinPHILA']
    s_id=0
    for tags in hashtags:
        tweets = []
        tweets_from_file =[]
        since_id=[]
        users=[]
        count=0
        print("Data Collected for:",tags)
        with open('collect.txt', 'a') as f:
            print("Data Collected for:",tags,file=f)

        tweets_from_file =open_file(tweets_from_file,tags)
        if tweets_from_file:
            for i in tweets_from_file:
                since_id.append(i['since_id'])
                if i['screen_name'] not in users:
                    users.append(i['screen_name'])
                    count+=1
            s_id=min(since_id)
            #print (s_id)
        #print("stating")
        #print ("length before search ",len(tweets_from_file))
        tweets = get_tweets(twitter,tags,tweets,s_id)
        tweets= select_required(tweets)
        for z in range(len(tweets)):
            if tweets[z]['screen_name'] not in users:
                users.append(tweets[z]['screen_name'])
                count+=1
        print("Total No. of Users who tweets for this hashtag:",count)
        with open('collect.txt', 'a') as f:
            print("Total No. of Users who tweets for this hashtag:",count,file=f)
        for i in tweets:
            tweets_from_file.append(i)
        write_file(tweets_from_file,tags)

    print('tweets saved to each hashtags file \n')

if __name__ == '__main__':
    main()
