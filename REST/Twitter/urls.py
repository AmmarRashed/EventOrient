from django.conf.urls import url
from .views import *

app_name = "Twitter"

urlpatterns = [
    url(r"^$", twitter_connections_reset, name="home"),
    url(r"^[R|r]aw/$", twitter_connections_raw, name="twitter_connections_raw"),
    url(r'^fb/$', twitter_fb_connections, name="twitter_fb_connections"),
    url(r'^api/fb', twitter_fb, name="twitter_fb"),
url(r'^api/twitter/(?P<degree_threshold>.+)/(?P<btw_threshold>.+)/(?P<pagerank_threshold>.+)/(?P<closeness_threshold>.+)/(?P<eigenvector_threshold>.+)/(?P<clust_threshold>.+)/(?P<date_index>.+)/(?P<recalculate_checked>.+)/(?P<recalculate_coms_checked>.+)/(?P<bidir>.+)/(?P<foci_checked>.+)/$',
        load_twitter_connections_json, name="load_twitter_connections_json")

]
# (?P<d>\w+)(?P<e>\w+)(?P<f>\w+)(?P<g>\w+)(?P<h>\w+)(?P<i>\w+)(?P<j>\w+)(?P<k>\w+)
