# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from functools import lru_cache

from django.http import JsonResponse
from django.shortcuts import render, render_to_response
import pandas as pd
from datetime import datetime
import unicodedata
from django.template import RequestContext
import psycopg2

from copy import deepcopy

import os, json

import networkx as nx
from networkx.readwrite import json_graph

_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(_dir)

user = "postgres"
password = "1_sehir_1"
dbname = "link_formation"
connection = psycopg2.connect('dbname=%s host=localhost user=%s password=%s'%(dbname, user, password))
# twitter_users = pd.read_sql("SELECT * FROM twitter_user", connection)
# twitter_users = twitter_users.where(twitter_users.match_name.str.len()>6)\
#                             .where(twitter_users.match_ratio>85)\
#                              .where(~twitter_users.name.str.contains("(?i)sehir"))\
#                              .dropna().set_index("id")

user_connections = pd.read_sql("SELECT * FROM twitter_connection", connection).drop('id', axis=1)
twitter_users = pd.read_csv(root_dir+ "/static/twitter_users.csv")


def clean(name):
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').lower().decode("ascii")

@lru_cache(maxsize=None)
def calculate_metrics(G):
    evc = nx.eigenvector_centrality(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    metrics = {"eigenvector_centrality": evc,
               "closeness_centrality": closeness,
               "betweenness": betweenness}
    return metrics


def homophily(nw, metric="lang"):
    langs_probs = dict()
    for n in nw.nodes():
        user = nw.nodes[n]
        langs_probs.setdefault(user[metric], 0)
        langs_probs[user[metric]] += 1
    heterogeneity_fraction_norm = 1 - sum([(float(i)/len(nw.nodes()))**2 for i in langs_probs.values()])
    cross_edges = sum([int(nw.nodes[f][metric] != nw.nodes[t][metric] ) for f,t in nw.edges()])
    return cross_edges/float(len(nw.edges())), heterogeneity_fraction_norm


def construct_network(connections):
    G = nx.DiGraph()
    for _, row in connections.iterrows():
        from_ = row["from_user_id"]
        to = row["to_user_id"]
        if from_ in twitter_users.tw_id and to in twitter_users.tw_id:
            G.add_edge(from_, to)

    augs = ["name", "screen_name", "match_name", "followers_count", "friends_count"]
    for node in G.nodes():
        user = twitter_users.loc[node]
        for aug in augs:
            if type(user[aug]) == str:
                m = clean(user[aug])
            else:
                m = user[aug]
            G.nodes[node][aug] = m

    evc = nx.eigenvector_centrality(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    metrics = {"eigenvector_centrality": evc,
               "closeness_centrality": closeness,
               "betweenness": betweenness,
               "pagerank": pagerank}
    for metric_name, metric in metrics.items():
        for ix, v in metric.items():
            G.nodes[ix][metric_name] = v
    return G


# twitter_connections_json = root_dir + "/static/networks/twitter_users_graph2.json"
# twitter_fb_j<son = root_dir + "/static/networks/twitter_fb.json"


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
date_index = 0
size_metric = "degree"
size_metrics = ["degree", "in_degree", "out_degree",
                "betweenness", "closeness_centrality", "eigenvector_centrality","pagerank","followers_count"]


def get_connections_by_date(date):
    nw = deepcopy(user_connections)
    for_col = nw.formation.apply(lambda x: date in x and x[date])
    return user_connections[for_col == True]

def get_dates():
    all_dates = set()
    str2date = lambda strdate: datetime.strptime(strdate, '%Y.%m.%d')  # 2018.05.08

    for dates in user_connections.formation.apply(lambda x: list(x)):
        for date in dates:
            all_dates.add(str2date(date))
    return [d.strftime('%Y.%m.%d') for d in sorted(all_dates)]


dates = get_dates()


def get_avg_metric(graph, metric):
    result = 0.0
    for node in graph["nodes"]:
        result += node[metric]
    return result/len(graph["nodes"])

def recalculate_metrics(nxg):
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



def present_in_date(changes_dates, queried_date):
    """
    checking if a connection is present in a queried date
    changes_dates: {d1:True, d2:False, d3:True} connection added or removed
    queried_date: e.g. "2018.05.08"
    """
    str2date = lambda strdate: datetime.strptime(strdate, '%Y.%m.%d')  # 2018.05.08
    changes = sorted(changes_dates,key=lambda d: str2date(d))
    queried_date = datetime.strptime(queried_date, '%Y.%m.%d')
    present = False
    for d in changes:
        if queried_date < str2date(d):
            break
        present = changes_dates[d]
    return present



def twitter_connections(request):

    global degree_threshold, filtered_twitter_connections, \
        btw_threshold, pagerank_threshold, closeness_threshold, \
        eigenvector_threshold, size_metric,size_metrics, recalculate_checked, \
        dates, date_index

    do_filter = False
    check = {"on":True, False:False}
    filtering = False

    if request.method == "POST":
        filtering = True
        degree_threshold = int(request.POST["degree_scroller"])
        btw_threshold = float(request.POST["btw_scroller"])
        pagerank_threshold = float(request.POST["pagerank_scroller"])
        closeness_threshold = float(request.POST["closeness_scroller"])
        eigenvector_threshold = float(request.POST["eigenvector_scroller"])
        date_index = int(request.POST["date"])
        recalculate_checked = check[request.POST.get("recalculate_metrics", False)]

        if date_index >= 0:
            for i in [degree_threshold, btw_threshold, pagerank_threshold, closeness_threshold, eigenvector_threshold]:
                if i*1 != 0:
                    do_filter = True
                    break

        size_metric = request.POST["size_metric"]
        if do_filter:
            if date_index > 0:
                connections = get_connections_by_date(dates[date_index])
                nw = construct_network(connections)
                data = nx.node_link_data(nw)
            else:
                data = deepcopy(twitter_connections_json)

            g = json_graph.node_link_graph(data, directed=True)

            filtered_twitter_connections = filter_by(g,
                                                     degree_threshold,
                                                     btw_threshold,
                                                     pagerank_threshold,
                                                     closeness_threshold,
                                                     eigenvector_threshold,
                                                     recalculate_checked,
                                                     directed=True)
        else:
            filtered_twitter_connections = deepcopy(twitter_connections_json)

    avgs = None
    if len(filtered_twitter_connections["nodes"])<1:
        sizes = [0]

    else:
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
               "recalculate_checked":int(recalculate_checked),
               "nodes_number": len(filtered_twitter_connections["nodes"]),
               "is_filtering":int(filtering),
               "dates":dates,
               "dates_dumped":json.dumps(list(dates)),
               "date_index":date_index}
    if avgs:
        context.update(avgs)

    return render(request, "twitter_connections.html", context)


@lru_cache(maxsize=None)
def filter_by(g, degree, btw, pagerank, closeness, eigenv, recalculate_node_metrics, directed=True):
    metrics = {"degree":degree, "betweenness":btw, "pagerank":pagerank,
               "closeness_centrality":closeness, "eigenvector_centrality":eigenv}
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
    if recalculate_node_metrics and len(c.nodes) != len(twitter_connections_json["nodes"]):
        return recalculate_metrics(c)
    return json_graph.node_link_data(c)


def load_twitter_connections_json(request):

    return JsonResponse(filtered_twitter_connections, safe=False)

def twitter_fb(request):
    return JsonResponse(twitter_fb_json, safe=False)

def twitter_fb_connections(request):
    return render_to_response("twitter_fb.html", {})
