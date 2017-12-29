from __future__ import print_function
import os
import sys
import ast
import json
import re
import string
import requests
import matplotlib.pyplot as plt
import threading
import Queue
import time
import requests_oauthlib
#import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import multiprocessing
from collections import Counter
from pyspark import SparkContext
from pyspark import SQLContext,Row
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.ml.feature import HashingTF,IDF, Tokenizer
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel

import threading
import Queue
import time

BATCH_INTERVAL = 10  # How frequently to update (seconds)
WINDOWS_LENGTH=60  #the duration of the window
SLIDING_INTERVAL=20 #the interval at which the window operation is performed
clusterNum=15 #Number of CLusters
sc = SparkContext(appName="cognitiveclass")
ssc = StreamingContext(sc, BATCH_INTERVAL)


def get_coord2(post):
    coord = tuple()
    try:
        if post['coordinates'] == None:
            coord = post['place']['bounding_box']['coordinates']
            coord = reduce(lambda agg, nxt: [agg[0] + nxt[0], agg[1] + nxt[1]], coord[0])
            coord = tuple(map(lambda t: t / 4.0, coord))
        else:
            coord = tuple(post['coordinates']['coordinates'])
    except TypeError:
        #print ('error get_coord')
        coord=(0,0)
    return coord

def get_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError, e:
    return False
  return json_object


def doc2vec(document):
    doc_vec = np.zeros(100)
    tot_words = 0

    for word in document:
        try:
            vec = np.array(lookup_bd.value.get(word))
            if vec!= None:
                doc_vec +=  vec
                tot_words += 1
        except:
            continue

    #return(tot_words)
    return doc_vec / float(tot_words)


def tokenize(text):
    tokens = []
    text = text.encode('ascii', 'ignore') #to decode
    text=re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) # to replace url with ''
    text = remove_spl_char_regex.sub(" ",text)  # Remove special characters
    text=text.lower()


    for word in text.split():
        if word not in stopwords \
            and word not in string.punctuation \
            and len(word)>1 \
            and word != '``':
                tokens.append(word)
    return tokens


def freqcount(terms_all):
    count_all = Counter()
    count_all.update(terms_all)
    return count_all.most_common(5)


# Create a DStream that will connect to hostname:port, like localhost:9999
dstream = ssc.socketTextStream("localhost", 9994)
#dstreamwin=dstream.window(WINDOWS_LENGTH, SLIDING_INTERVAL)
dstream_tweets=dstream.map(lambda post: get_json(post))\
     .filter(lambda post: post != False)\
     .filter(lambda post: 'created_at' in post)\
     .map(lambda post: (get_coord2(post)[0],get_coord2(post)[1],post["text"]))\
     .filter(lambda tpl: tpl[0] != 0)\
     .filter(lambda tpl: tpl[2] != '')\
     .map(lambda tpl: (tpl[0],tpl[1],tokenize(tpl[2])))\
     .map(lambda tpl:(tpl[0],tpl[1],tpl[2],doc2vec(tpl[2])))
#dstream_tweets.pprint()


trainingData=dstream_tweets.map(lambda tpl: [tpl[0],tpl[1]]+tpl[3].tolist())
#trainingData.pprint()
testdata=dstream_tweets.map(lambda tpl: (([tpl[0],tpl[1]],tpl[2]),[tpl[0],tpl[1]]+tpl[3].tolist()))

ssc.start()
ssc.awaitTermination()
