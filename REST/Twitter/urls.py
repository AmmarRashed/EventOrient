from django.conf.urls import url
from .views import *

app_name = "Twitter"
urlpatterns = [
    url(r'^$', twitter_connections, name="twitter_connections")
]