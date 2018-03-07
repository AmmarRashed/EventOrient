# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.http import JsonResponse
from django.shortcuts import render, render_to_response
import os, json

_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(_dir)
# twitter_connections_json = root_dir + "/static/networks/twitter_users_graph2.json"
# twitter_fb_json = root_dir + "/static/networks/twitter_fb.json"

twitter_connections_json = json.load(open(root_dir+ "/static/networks/twitter_users_graph2.json","r"))
twitter_fb_json = json.load(open(root_dir + "/static/networks/twitter_fb.json", "r"))

def twitter_connections(request):
    return render_to_response("twitter_connections.html", {'twitter_connections_json':json.dumps(twitter_connections_json)})


def load_twitter_connections_json(request):
    return JsonResponse(twitter_connections_json, safe=False)

def twitter_fb(request):
    return JsonResponse(twitter_fb_json, safe=False)

def twitter_fb_connections(request):
    return render_to_response("twitter_fb.html", {})
