# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from django.shortcuts import render, get_object_or_404
from django.http import Http404,HttpResponse

def twitter_connections(request):
    return render(request, "twitter_connections.html", {})