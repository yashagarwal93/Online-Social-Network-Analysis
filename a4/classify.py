"""
classify.py
"""
import pickle
from pathlib import Path
import sys
import re
import numpy as np
import time
from collections import Counter
from TwitterAPI import TwitterAPI
from collections import Counter, defaultdict, deque
import csv
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from collect import open_file



def get_tweets(file,tweet_list):
    for i in file:
        t=[]
        tweet_list.append(i['tweet'])
        #screen_name.append(i['userid'])
    return tweet_list



def process_Data(allTweet):
    data = []
    for json in allTweet:
        tweet_data = []
        tweet_data.append(json['username'])
        tweet_data.append(json['userid'])
        tweet_data.append(json['description'])
        tweet_data.append(json['tweet'])

        data.append(tweet_data)
    return data


def write_tweets_csv(data):
    with open('twitter_data.csv', 'w',encoding='utf-8') as fp:
        a = csv.writer(fp)
        a.writerows(data)





def write_classification(f,logistic_prediction,tweets_list):

    tweet_classified_labelling = []
    for i in range(len(tweets_list)):
        tweet_classified_labelling.append((logistic_prediction[i],tweets_list[i]))

    with open(f, 'w',encoding='utf-8',newline='') as fp1:
        filewriter = csv.writer(fp1)
        filewriter.writerows(tweet_classified_labelling)


def read_tarining_data(filename):
    tweets = []
    labels1 = []
    with open(filename, 'r',encoding='utf-8') as x:
        filereader = csv.reader(x)
        for  row in filereader:
            labels1.append(row[0])
            tweets.append(row[1])
    return tweets,np.array(labels1)


def tokenize(text):

    tokens = re.findall(r"\w+|\S", text.lower(),flags = re.L)
    tokens1 = []
    for i in tokens:
        i = re.sub('http\S+', 'THIS_IS_A_URL', i)
        i = re.sub('@\S+', 'THIS_IS_A_MENTION', i)
        x = re.findall(r"\w+|\S", i,flags = re.U)
        for j in x:
            tokens1.append(j)
    return tokens1


def do_vectorize(tokenizer_fn=tokenize, min_df=1,
                 max_df=1., binary=False, ngram_range=(1,1)):


    vectorizer = CountVectorizer(input = 'content', tokenizer = tokenizer_fn, min_df=min_df,
                                     max_df=max_df, binary=True, ngram_range=ngram_range,
                                 dtype = 'int',analyzer='word',token_pattern='(?u)\b\w\w+\b',encoding='utf-8' )
    return vectorizer


def get_clf():
    return LogisticRegression()


def prediction(CLF,trained_CSR,trained_label,untrained_tweets_CSR):
    CLF.fit(trained_CSR,trained_label)
    predicted = CLF.predict(untrained_tweets_CSR)
    return predicted



def do_cross_validation(X, y,clf, k=5):
    cv = KFold(len(y), k)
    acc = []

    for train_idx, test_idx in cv:
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        accuracy = accuracy_score(y[test_idx], predicted)
        acc.append(accuracy)

    avg = np.mean(acc)
    return avg



def neg_no_of_user(logistic_prediction,data):

    list_neg=[]
    for i in range(len(logistic_prediction)):
        if(logistic_prediction[i]=='-1'):
            if (data[i][1] not in list_neg):
                list_neg.append(data[i][1])
    return(len(list_neg))

def pos_no_of_user(logistic_prediction,data):
    list_pos=[]

    for i in range(len(logistic_prediction)):
        if(logistic_prediction[i]=='1'):
            if (data[i][1] not in list_pos):
                list_pos.append(data[i][1])
    return(len(list_pos))


def main():
    hashtags=['#Celtics','#DetroitBasketball','#DefendTheLand','#WeTheNorth','#MADEinPHILA']
    #hashtags=['#DubNation']
    tweets_list=[]
    text_file = open("classify.txt" , "w")
    labeled_tweets,labels = read_tarining_data('tweet_manual_labelling.csv')
    clf_logistic = get_clf()
    #get the vectorizer object
    vectorizer = do_vectorize()
    X = vectorizer.fit_transform(tweet for tweet in labeled_tweets)
    y = np.array(labels)
    logistic_regression_accuracy = (do_cross_validation(X, y,clf_logistic))*100
    print('Average cross validation accuracy for Logistic Regression=%.1f percentage' % (logistic_regression_accuracy))
    with open('classify.txt', 'a') as f:
        print('Average cross validation accuracy for Logistic Regression=%.1f percentage' % (logistic_regression_accuracy),file=f)

    for tags in hashtags:
        tweets=[]
        result=[]
        tweet_list=[]
        screen_name=[]
        neg_unique_user=0
        pos_unique_user=0
        #get tweets for each team
        tweets= open_file(tweets,tags)
        #to store in csv process the data
        data=process_Data(tweets)
        write_tweets_csv(data)
        #data to be tested
        tweets_list= get_tweets(tweets,tweet_list)
        #print(data[733])
        #create_for_manual(tweets_list,afinn) only one -- already done and provided in  for you
        #Prediction for unlabelled Tweets
        test_tweet_vector = vectorizer.transform(t for t in tweets_list)
        logistic_prediction =prediction(clf_logistic,X,y,test_tweet_vector)
        fname=tags+ '.csv'
        write_classification(fname,logistic_prediction,tweets_list)
        #print(logistic_prediction[2])
        result = dict(Counter(logistic_prediction))
        pos_unique_user=pos_no_of_user(logistic_prediction,data)
        neg_unique_user=neg_no_of_user(logistic_prediction,data)

        #print(result[user])

        print ("Logistic Regression Results for team",tags)
        with open('classify.txt', 'a') as f:
            print ("Logistic Regression Results for team",tags,file=f)

        for i in result:
            if i == '-1':
                print ("\t Total Number of Tweets aganist team\t\t\t\t",result[i])
                print ("\t No.of Unique users who tweeted aganist team\t\t\t",neg_unique_user)
                with open('classify.txt', 'a') as f:
                    print ("\t Total Number of Tweets aganist team\t\t\t\t",result[i],file=f)
                    print ("\t No.of Unique users who tweeted aganist team\t\t\t",neg_unique_user,file=f)

                #print(neg_unique_user)



            elif i == '1':
                print ("\t Number of Tweets supporting team \t\t\t\t%d" %result[i])
                print ("\t No.of Unique users who tweeted supporting the team\t\t",pos_unique_user)
                with open('classify.txt', 'a') as f:
                    print ("\t Number of Tweets supporting team \t\t\t\t%d" %result[i],file=f)
                    print ("\t No.of Unique users who tweeted supporting the team\t\t",pos_unique_user,file=f)


        labeled_tweets,labels=read_tarining_data(fname)

if __name__ == '__main__':
    main()
