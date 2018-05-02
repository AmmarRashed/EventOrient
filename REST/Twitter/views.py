# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.http import JsonResponse
from django.shortcuts import render, render_to_response
from django.template import RequestContext
from copy import deepcopy

import os, json
from copy import deepcopy

import networkx as nx
from networkx.readwrite import json_graph

_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(_dir)
# twitter_connections_json = root_dir + "/static/networks/twitter_users_graph2.json"
# twitter_fb_json = root_dir + "/static/networks/twitter_fb.json"

twitter_connections_json = json.load(open(root_dir+ "/static/networks/twitter_users_graph2.json","r"))
twitter_fb_json = json.load(open(root_dir + "/static/networks/twitter_fb.json", "r"))


filtered_twitter_connections = deepcopy(twitter_connections_json)


degree = 0

def twitter_connections(request):

    # return render_to_response("twitter_connections.html",{}, RequestContext(request))
    global degree, filtered_twitter_connections
    if request.method == "POST":
        degree = int(request.POST["degree_scroller"])
        filtered_twitter_connections = filter_by(twitter_connections_json, degree, True)
    degrees = [n["degree"] for n in filtered_twitter_connections["nodes"]]
    return render(request,"twitter_connections.html",{"current_degree":degree,
                                                      "minDegree":min(degrees),
                                                      "maxDegree":max(degrees),
                                                      "avgDegree":(sum(degrees)/float(len(degrees)))})


def filter_by(data_, degree, directed):
    g = json_graph.node_link_graph(data_, directed=True)
    if directed:
        try:
            g = g.to_directed()
        except:
            pass
    c = g.copy()
    for node in g.nodes():
        if g.nodes[node]["degree"] < degree:
            c.remove_node(node)
    return json_graph.node_link_data(c)


# def filter_by2(data, degree, directed):
#     selected_nodes_ids = {node["id"]:True for node in data["nodes"] if node["degree"]>= degree}
#     selected_nodes = [node for node in data["nodes"] if node["degree"]>= degree]
#     selected_edges = [edge for edge in data["links"]
#                       if edge["source"] in selected_nodes_ids or
#                          edge["target"] in selected_nodes_ids]
#     data["nodes"] = selected_nodes
#     data["links"] = selected_edges
#     data["directed"] = directed
#     return data


def load_twitter_connections_json(request):

    return JsonResponse(filtered_twitter_connections, safe=False)

def twitter_fb(request):
    return JsonResponse(twitter_fb_json, safe=False)

def twitter_fb_connections(request):
    return render_to_response("twitter_fb.html", {})
