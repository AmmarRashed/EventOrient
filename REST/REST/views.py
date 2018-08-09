# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from django.shortcuts import render, get_object_or_404
from django.http import Http404,HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView


class Home(APIView):
    def get(self, request):
        return Response("Hello")

    def post(self):
        pass


def home(request):
    return HttpResponse("Home page")
