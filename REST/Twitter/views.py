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
recalculate_checked = 1
size_metric = "degree"
size_metrics = ["degree", "in_degree", "out_degree",
                "betweenness", "closeness_centrality", "eigenvector_centrality","pagerank","followers_count"]

def get_avg_metric(graph, metric):
    result = 0.0
    for node in graph["nodes"]:
        result += node[metric]
    return result/len(graph["nodes"])

def recalculate_metrics(filtered_twitter_connections):
    nxg = json_graph.node_link_graph(filtered_twitter_connections, directed=True)
    for ix, deg in nxg.degree(nxg.nodes()):
        nxg.node[ix]['degree'] = deg

    for ix, in_deg in nxg.in_degree(nxg.nodes()):
        nxg.node[ix]['in_degree'] = in_deg

    for ix, out_deg in nxg.out_degree(nxg.nodes()):
        nxg.node[ix]['out_degree'] = out_deg

    evc = nx.eigenvector_centrality(nxg)
    closeness = nx.closeness_centrality(nxg)
    betweenness = nx.betweenness_centrality(nxg)
    pagerank = nx.pagerank(nxg)
    cntr_metrics = {"eigenvector_centrality": evc,
                    "closeness_centrality": closeness,
                    "betweenness": betweenness,
                    "pagerank": pagerank}

    for metric_name, metric in cntr_metrics.items():
        for ix, v in metric.items():
            nxg.node[ix][metric_name] = v
    return json_graph.node_link_data(nxg)


def twitter_connections(request):

    global degree_threshold, filtered_twitter_connections, \
        btw_threshold, pagerank_threshold, closeness_threshold,\
        eigenvector_threshold, size_metric,size_metrics, recalculate_checked

    do_filter = False
    check = {"on":True, False:False}
    if request.method == "POST":
        degree_threshold = int(request.POST["degree_scroller"])
        btw_threshold = float(request.POST["btw_scroller"])
        pagerank_threshold = float(request.POST["pagerank_scroller"])
        closeness_threshold = float(request.POST["closeness_scroller"])
        eigenvector_threshold = float(request.POST["eigenvector_scroller"])
        recalculate_checked = check[request.POST.get("recalculate_metrics", False)]

        for i in [degree_threshold, btw_threshold, pagerank_threshold, closeness_threshold, eigenvector_threshold]:
            if int(i + 1) != 1:
                do_filter = True
                break

        size_metric = request.POST["size_metric"]
        if do_filter:
            filtered_twitter_connections = filter_by(twitter_connections_json,
                                                 degree_threshold,
                                                 btw_threshold,
                                                 pagerank_threshold,
                                                 closeness_threshold,
                                                 eigenvector_threshold, directed=True)
        else:
            filtered_twitter_connections = deepcopy(twitter_connections_json)

    avgs = None
    if len(filtered_twitter_connections["nodes"])<1:
        sizes = [0]
    else:
        if do_filter and recalculate_checked:
            filtered_twitter_connections=recalculate_metrics(filtered_twitter_connections)
        avgs = {"avg_"+metric:get_avg_metric(filtered_twitter_connections, metric) for metric in size_metrics}
        sizes = [n[size_metric] for n in filtered_twitter_connections["nodes"]]

    context = {"degree_threshold": degree_threshold,
               "minSize": min(sizes),
               "maxSize": max(sizes),
               "avgSize": (sum(sizes) / float(len(sizes))),
               "btw_threshold": btw_threshold,
               "pagerank_threshold": pagerank_threshold,
               "closeness_threshold": closeness_threshold,
               "eigenvector_threshold": eigenvector_threshold,
               "size_metric": size_metric,
               "size_metrics": size_metrics,
               "recalculate_checked":int(recalculate_checked)}
    if avgs:
        context.update(avgs)

    return render(request, "twitter_connections.html", context)


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


def load_twitter_connections_json(request):

    return JsonResponse(filtered_twitter_connections, safe=False)

def twitter_fb(request):
    return JsonResponse(twitter_fb_json, safe=False)

def twitter_fb_connections(request):
    return render_to_response("twitter_fb.html", {})
