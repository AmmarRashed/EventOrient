import os

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import json

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.2 pyspark-shell'
consumer_key="9tXctu2Bsh3nLH3RqqmdPCKBk"
consumer_secret = "AV1kQou6NmkXBMQqXVqPwE3iUxQqt4uXAF5VEg80x7ORtKC7is"
access_token_key = "291122559-9NXOCoI49lcGnCJTDsaAuBXKzf8xuNm0W5yuAyt9"
access_token_secret = "	cXJaW18DGBy8wvJHVc9HRpvETUwhloQt5GUYHdAnwKOJq"

sc = SparkContext(appName="PythonSparkStreamingKafka_RM_01")
sc.setLogLevel("WARN")
ssc = StreamingContext(sc, 60)
kafkaStream = KafkaUtils.createStream(ssc, 'cdh57-01-node-01.moffatt.me:2181', 'spark-streaming', {'twitter':1})
parsed = kafkaStream.map(lambda v: json.loads(v[1]))
parsed.count().map(lambda x:'Tweets in this batch: %s' % x).pprint()
authors_dstream = parsed.map(lambda tweet: tweet['user']['screen_name'])
author_counts = authors_dstream.countByValue()
author_counts.pprint()
author_counts_sorted_dstream = author_counts.transform((lambda foo:foo.sortBy(lambda x:( -x[1]))))
author_counts_sorted_dstream.pprint()
top_five_authors = author_counts_sorted_dstream.transform(lambda rdd:sc.parallelize(rdd.take(5)))
top_five_authors.pprint()
ssc.start()
ssc.awaitTermination()