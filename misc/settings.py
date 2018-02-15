# Sign in and create an app on https://dev.twitter.com.
# You will get the consumer_key and consumer_secret. 
CONSUMER_KEY="9tXctu2Bsh3nLH3RqqmdPCKBk"
CONSUMER_SECRET="AV1kQou6NmkXBMQqXVqPwE3iUxQqt4uXAF5VEg80x7ORtKC7is"

# After create and get the above key, you will be redirected to app page.
# Create an access token under the the "Your access token" section, then you'll get the following two key
ACCESS_TOKEN="291122559-9NXOCoI49lcGnCJTDsaAuBXKzf8xuNm0W5yuAyt9"
ACCESS_TOKEN_SECRET="cXJaW18DGBy8wvJHVc9HRpvETUwhloQt5GUYHdAnwKOJq"

#Create an account at GeoNames and enable the account for use of free web services
#http://www.geonames.org/login
GEONAMES_USERNAME="ammarrashed"

# Delimiter for the SQL output
DELIMITER = u"|"

# set to a database URI according to sqlalchemy's scheme for the database to dump stored tweets into
DATABASE_URI = "postgresql://"
# to send parameters to the connection argument, set the DATABASE_URI to
# the database protocol you want (e.g. "postgresql://") and then uncomment
# and fill in the following DATABASE_CONFIG
#
DATABASE_CONFIG = {
    'database':'sehir',
    'host':'localhost',
    'user':'postgres',
    'password':'1_sehir_1'
}

# Running in debug mode, the system prints a lot more information
DEBUG = True

# set to a database URI according to sqlalchemy's scheme for the database to allow 
# various operators use as scratch space
SCRATCHSPACE_URI = "postgresql://scratch.db"
# what prefix should tables used for scratchspace get
SCRATCHSPACE_PREFIX = "tweeql_scratch__"
