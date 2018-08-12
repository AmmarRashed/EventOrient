from django.conf.urls import url
from .views import *

app_name = "Twitter"

urlpatterns = [
    url(r"^$", twitter_connections, name="home"),
    url(r"^[T|t]witter/$", twitter_connections, name="twitter_connections"),
    url(r'^fb/$', twitter_fb_connections, name="twitter_fb_connections"),
    url(r'^api/fb', twitter_fb, name="twitter_fb"),
    url(r'^api/twitter/$', load_twitter_connections_json, name="load_twitter_connections_json")
]