import socket
import sys
import requests
import requests_oauthlib
import json

CONSUMER_KEY = "9tXctu2Bsh3nLH3RqqmdPCKBk"
CONSUMER_SECRET = "AV1kQou6NmkXBMQqXVqPwE3iUxQqt4uXAF5VEg80x7ORtKC7is"
ACCESS_TOKEN = "291122559-9NXOCoI49lcGnCJTDsaAuBXKzf8xuNm0W5yuAyt9"
ACCESS_SECRET = "cXJaW18DGBy8wvJHVc9HRpvETUwhloQt5GUYHdAnwKOJq"

DEFAULT_AUTH = requests_oauthlib.OAuth1(CONSUMER_KEY, CONSUMER_SECRET,ACCESS_TOKEN, ACCESS_SECRET)

DEFAULT_QUERY_DATA = [('language', 'en'), ('locations', '-130,-20,100,50'),('track','#')]
DEFAULT_URL = 'https://stream.twitter.com/1.1/statuses/filter.json'


def get_tweets(url=DEFAULT_URL, query_data=DEFAULT_QUERY_DATA, auth=DEFAULT_AUTH, print_=True):
    query_url = url + '?' + '&'.join([str(t[0]) + '=' + str(t[1]) for t in query_data])
    response = requests.get(query_url, auth=auth, stream=True)
    if print_:
        print(query_url, response)
    return response

def send_tweets_to_spark(http_resp, tcp_connection, tweet_parameters=['text', 'languages']):
    for line in http_resp.iter_lines():
        try:
            full_tweet = json.loads(line)
            package = ' '.join([str(full_tweet[parameter]) for parameter in tweet_parameters]) + '\n'
            print (full_tweet['text'])
            print ('-'*10)
            tcp_connection.send(package)
        except:
            e = sys.exc_info()[0]
            print("Error: %s" % e)


TCP_IP = "localhost"
TCP_PORT = 9999
conn = None
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((TCP_IP, TCP_PORT))
except:
    pass
s.listen(1)
print("Waiting for TCP connection...")
conn, addr = s.accept()
print("Connected... Starting getting tweets.")
resp = get_tweets()
send_tweets_to_spark(resp, conn)
