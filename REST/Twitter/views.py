# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.http import JsonResponse
from django.shortcuts import render, render_to_response
from django.template import RequestContext
from copy import deepcopy

import os, json

import networkx as nx
from networkx.readwrite import json_graph

_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(_dir)
# twitter_connections_json = root_dir + "/static/networks/twitter_users_graph2.json"
# twitter_fb_json = root_dir + "/static/networks/twitter_fb.json"


twitter_connections_path = root_dir+ "/static/networks/twitter_users_graph2.json"
twitter_connections_json = json.load(open(twitter_connections_path,"r"))

twitter_fb_path = root_dir + "/static/networks/twitter_fb.json"
twitter_fb_json = json.load(open(twitter_fb_path, "r"))


filtered_twitter_connections = deepcopy(twitter_connections_json)


degree_threshold = 0
btw_threshold = 0.0
pagerank_threshold = 0.0
closeness_threshold = 0.0
eigenvector_threshold = 0.0
size_metric = "degree"
size_metrics = ["degree", "in_degree", "out_degree",
                "betweenness", "closeness_centrality", "eigenvector_centrality","pagerank","followers_count"]
def twitter_connections(request):

    # return render_to_response("twitter_connections.html",{}, RequestContext(request))
    global degree_threshold, filtered_twitter_connections, \
        btw_threshold, pagerank_threshold, closeness_threshold,\
        eigenvector_threshold, size_metric,size_metrics
    if request.method == "POST":
        degree_threshold = int(request.POST["degree_scroller"])
        btw_threshold = float(request.POST["btw_scroller"])
        pagerank_threshold = float(request.POST["pagerank_scroller"])
        closeness_threshold = float(request.POST["closeness_scroller"])
        eigenvector_threshold = float(request.POST["eigenvector_scroller"])
        size_metric = request.POST["size_metric"]
        print("size_metric", size_metric)
        filtered_twitter_connections = filter_by(twitter_connections_json,
                                                 degree_threshold,
                                                 btw_threshold,
                                                 pagerank_threshold,
                                                 closeness_threshold,
                                                 eigenvector_threshold, directed=True)

    sizes = [n[size_metric] for n in filtered_twitter_connections["nodes"]]
    if len(filtered_twitter_connections["nodes"])<1:
        sizes = [0]
    return render(request, "twitter_connections.html", {"degree_threshold":degree_threshold,
                                                      "minDegree":min(sizes),
                                                      "maxDegree":max(sizes),
                                                      "avgDegree":(sum(sizes)/float(len(sizes))),
                                                      "btw_threshold":btw_threshold,
                                                      "pagerank_threshold":pagerank_threshold,
                                                      "closeness_threshold":closeness_threshold,
                                                      "eigenvector_threshold":eigenvector_threshold,
                                                      "size_metric":size_metric,
                                                      "size_metrics":size_metrics})


def filter_by(data_, degree, btw, pagerank, closeness, eigenv, directed=True):
    metrics = {"degree":degree, "betweenness":btw, "pagerank":pagerank,
               "closeness_centrality":closeness, "eigenvector_centrality":eigenv}
    g = json_graph.node_link_graph(data_, directed=directed)
    if directed:
        try:
            g = g.to_directed()
        except:
            pass
    c = g.copy()
    for node in g.nodes():
        invalid = False
        for metric in metrics:
            threshold = metrics[metric]
            if g.nodes[node][metric] < threshold:
                invalid = True
                break
        if invalid:
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
