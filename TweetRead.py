# TweetRead.py
# This first python script doesn’t use Spark at all:


from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import pickle
MY_TWITTER_ID = "291122559"
sehirians_twitter_ids = pickle.load(open("datasets/sehirians_twitter_ids.json","rb"))+[MY_TWITTER_ID]

# consumer_key = os.environ['TWITTER_CONSUMER_KEY']
# consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
# access_token = os.environ['TWITTER_ACCESS_TOKEN']
# access_secret = os.environ['TWITTER_ACCESS_SECRET']

consumer_key = "9tXctu2Bsh3nLH3RqqmdPCKBk"
consumer_secret = "AV1kQou6NmkXBMQqXVqPwE3iUxQqt4uXAF5VEg80x7ORtKC7is"
access_token = "291122559-9NXOCoI49lcGnCJTDsaAuBXKzf8xuNm0W5yuAyt9"
access_secret = "cXJaW18DGBy8wvJHVc9HRpvETUwhloQt5GUYHdAnwKOJq"



class TweetsListener(StreamListener):
    def __init__(self, csocket):
        self.client_socket = csocket

    def on_data(self, data):
        try:
            print(data.split('\n'))
            self.client_socket.send(data)
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True


def sendData(c_socket):
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    twitter_stream = Stream(auth, TweetsListener(c_socket))
    twitter_stream.filter(follow=sehirians_twitter_ids)


if __name__ == "__main__":
    s = socket.socket()  # Create a socket object
    host = "localhost"  # Get local machine name
    port = 8000  # Reserve a port for your service.
    try:
        s.bind((host, port))  # Bind to the port
    except OSError:
        pass


    print("Listening on port: %s" % str(port))

    s.listen(5)  # Now wait for client connection.
    c, addr = s.accept()  # Establish connection with client.

    print("Received request from: " + str(addr))

    sendData(c)