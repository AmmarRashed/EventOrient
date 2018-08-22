# -*- coding: utf-8 -*-
from __future__ import unicode_literals

try:
    from functools import lru_cache
except ImportError:
    from repoze.lru import lru_cache

from django.http import JsonResponse
from django.shortcuts import render, render_to_response
import pandas as pd
from datetime import datetime
import unicodedata
from django.template import RequestContext
# import psycopg2

from copy import deepcopy

import os, json

import networkx as nx
import snap
from networkx.readwrite import json_graph

_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(_dir)

user = "postgres"
password = "1_sehir_1"
dbname = "link_formation"
# connection = psycopg2.connect('dbname=%s host=localhost user=%s password=%s'%(dbname, user, password))
# twitter_users = pd.read_sql("SELECT * FROM twitter_user", connection)
# twitter_users = twitter_users.where(twitter_users.match_name.str.len()>6)\
#                             .where(twitter_users.match_ratio>85)\
#                              .where(~twitter_users.name.str.contains("(?i)sehir"))\
#                              .dropna().set_index("id")

# user_connections = pd.read_sql("SELECT * FROM twitter_connection", connection).drop('id', axis=1)
user_connections = pd.read_csv(root_dir+ "/static/filtered_twitter_connections.csv")
import ast
str2dict = lambda d : ast.literal_eval(d)
user_connections.formation = user_connections.formation.apply(str2dict)


twitter_users = pd.read_csv(root_dir+ "/static/filtered_twitter_users.csv", index_col="id")
orgs = set(twitter_users[twitter_users.is_org].truncated_id)


def clean(name):
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').lower().decode("ascii")


def networkx_to_snappy(nxg, directed=False):
    if directed:
        g = snap.PNGraph.New()
    else:
        g = snap.PUNGraph.New()

    for n in nxg.nodes():
        g.AddNode(n)
    for f, t in nxg.edges():
        g.AddEdge(f, t)

    return g


def get_homophily(nw, metric="lang"):
    langs_probs = dict()
    for n in nw.nodes():
        user = nw.nodes[n]
        langs_probs.setdefault(user[metric], 0)
        langs_probs[user[metric]] += 1
    heterogeneity_fraction_norm = 1 - sum(
        [(float(i)/len(nw.nodes()))**2 for i in langs_probs.values()])
    cross_edges = sum(
        [int(nw.nodes[f][metric] != nw.nodes[t][metric] ) for f,t in nw.edges()])
    cross_metric_ratio = cross_edges/float(len(nw.edges()))
    print("cross-metric edges ratio: ", cross_metric_ratio)
    print("Heterogeneity Fraction Norm", heterogeneity_fraction_norm)
    return cross_metric_ratio < heterogeneity_fraction_norm, cross_metric_ratio, heterogeneity_fraction_norm


def get_bidir_edges(G):
    bidir_edges = 0
    for f,t in deepcopy(G.edges):
        if G.has_edge(t, f):
            bidir_edges += 1

        elif bidir:
            G.remove_edge(f, t)
    return G, bidir_edges


def label_nodes_SCCs(G):
    nodes_sccs = {}  # {node: scc_id}
    snappy_directed = networkx_to_snappy(G, True)
    components = snap.TCnComV()
    sccs = snap.GetSccs(snappy_directed, components)

    for CnCom in components:
        for n in CnCom:
            nodes_sccs[n] = CnCom


    for node in G.nodes():
        m = str(nodes_sccs[node][0])
        G.nodes[node]["SCC"] = m

    return G


def label_nodes_communities(G, recalculate=False):
    if recalculate:

        g = networkx_to_snappy(G)
        CmtyV = snap.TCnComV()
        modularity = snap.CommunityGirvanNewman(g, CmtyV)
        nodes_communities = {}  # {node: [community]}
        for i, Cmty in enumerate(CmtyV):
            for NI in Cmty:
                nodes_communities.setdefault(NI, [])
                nodes_communities[NI].append(i + 2)

    for node in G.nodes():
        try:
            user = twitter_users.loc[node]
            m = str(user['community'])
        except KeyError:
            m = "0"
        if recalculate and 'foci' not in m:
            m = str(nodes_communities[node][0])
        G.nodes[node]["community"] = m
    return G

def construct_network(connections, rec_metrics=True, include_foci=True, recalculate_coms=False, recalculate_SCCs=True):
    G = nx.DiGraph()
    truncate = lambda x: int(str(int(x))[:9])
    for _, row in connections.iterrows():
        f = row["from_user_id"]
        t = row["to_user_id"]
        from_ = truncate(f)
        to = truncate(t)
        if not include_foci and (from_ in orgs or to in orgs):
            continue
        if from_ in twitter_users.truncated_id and to in twitter_users.truncated_id:
            G.add_edge(from_, to)
    new_G, bidir_edges = get_bidir_edges(G)


    # augs = ["name", "screen_name", "match_name", "followers_count", "friends_count", "lang"]
    # getting pre-calculated communities >> THIS IS JUST OPTIONAL
    augs = ["name", "screen_name", "match_name", "followers_count", "friends_count", "lang"]

    G = label_nodes_communities(G, recalculate = recalculate_coms)

    if recalculate_SCCs:
        G = label_nodes_SCCs(G)



    for node in G.nodes():
        user = twitter_users.loc[node]
        for aug in augs:
            G.nodes[node][aug] = user[aug]

    return recalculate_metrics(G, parse=False, centralities=rec_metrics), bidir_edges



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
clust_threshold = 0.0
recalculate_checked = 0

DEFAULT_RECALCULATE_COMMUNITIES = False
DEFAULT_FOCI_CHECKED = 1
DEFAULT_BIDIR = 0

foci_checked = DEFAULT_FOCI_CHECKED
bidir = DEFAULT_BIDIR
recalculate_coms_checked = DEFAULT_RECALCULATE_COMMUNITIES

recalculate_SCCs_checked = True

date_index = 0
size_metric = "degree"
size_metrics = ["degree", "in_degree", "out_degree",
                "betweenness", "closeness_centrality", "eigenvector_centrality",
                "pagerank","clustering_coefficient", "followers_count"]


@lru_cache(maxsize=None)
def get_connections_by_date(date):
    nw = deepcopy(user_connections)
    for_col = nw.formation.apply(lambda dates: present_in_date(dates,date))
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
        try:
            result += node[metric]
        except KeyError:
            return 0.
    return result/len(graph["nodes"])


@lru_cache(maxsize=None)
def recalculate_metrics(nxg, parse=True, centralities=True):
    nxg = nxg.to_directed()
    for ix, deg in nxg.degree(nxg.nodes()):
        nxg.node[ix]['degree'] = deg
    for ix, in_deg in nxg.in_degree(nxg.nodes()):
        nxg.node[ix]['in_degree'] = in_deg

    for ix, out_deg in nxg.out_degree(nxg.nodes()):
        nxg.node[ix]['out_degree'] = out_deg
    if centralities:
        try:
            evc = nx.eigenvector_centrality(nxg, max_iter=200)
            closeness = nx.closeness_centrality(nxg)
            betweenness = nx.betweenness_centrality(nxg)
            pagerank = nx.pagerank(nxg)
            un_nxg = nxg.to_undirected()
            clustering = nx.clustering(un_nxg)
            cntr_metrics = {"eigenvector_centrality": evc,
                            "closeness_centrality": closeness,
                            "betweenness": betweenness,
                            "pagerank": pagerank,
                            "clustering_coefficient":clustering}

            for metric_name, metric in cntr_metrics.items():
                for ix, v in metric.items():
                    nxg.node[ix][metric_name] = v
        except:
            pass
    if parse:
        data = json_graph.node_link_data(nxg)
        # with open(root_dir + "/static/networks/latest_tw_ntw.json", 'w') as f:
        #     json.dump(data, f, indent=4)
        # return data
    return nxg



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
        eigenvector_threshold, size_metric,size_metrics, recalculate_checked, foci_checked,\
        dates, date_index, clust_threshold, bidir, recalculate_coms_checked, recalculate_SCCs_checked

    check = {"on":True, False:False}
    filtering = False

    if request.method == "POST":
        filtering = True
        degree_threshold = int(request.POST["degree_scroller"])
        btw_threshold = float(request.POST["btw_scroller"])
        pagerank_threshold = float(request.POST["pagerank_scroller"])
        closeness_threshold = float(request.POST["closeness_scroller"])
        eigenvector_threshold = float(request.POST["eigenvector_scroller"])
        clust_threshold = float(request.POST["clust_scroller"])
        date_index = int(request.POST["date"])
        recalculate_checked = check[request.POST.get("recalculate_metrics", False)]
        recalculate_coms_checked = check[request.POST.get("recalculate_coms_checked", False)]
        recalculate_SCCs_checked = check[request.POST.get("recalculate_SCCs_checked", "on")]

        foci_checked = check[request.POST.get("include_foci", False)]
        bidir = check[request.POST.get("bidir", False)]

        do_filter = foci_checked != DEFAULT_FOCI_CHECKED or bidir != DEFAULT_BIDIR
        if not do_filter:
            for i in [date_index, degree_threshold, btw_threshold, pagerank_threshold,
                      closeness_threshold, eigenvector_threshold, clust_threshold]:
                if i != 0:
                    do_filter = True
                    break
        size_metric = request.POST["size_metric"]

        if do_filter:
            # if date_index == 0 and foci_checked==DEFAULT_FOCI_CHECKED and bidir==DEFAULT_BIDIR:  # same network
            #     data = deepcopy(twitter_connections_json)
            # else:
            cons = get_connections_by_date(dates[date_index])
            nw, bidir_edges = construct_network(cons, recalculate_checked, foci_checked,
                                                recalculate_coms_checked, recalculate_SCCs_checked)
            data = nx.node_link_data(nw)
            with open('temp2.json', 'w') as f:
                print(len(data["nodes"]), "SIZE")
                json.dump(data, f, indent=4)
            g = json_graph.node_link_graph(data, directed=True)

            filtered_twitter_connections = filter_by(g,
                                                     degree_threshold,
                                                     btw_threshold,
                                                     pagerank_threshold,
                                                     closeness_threshold,
                                                     eigenvector_threshold,
                                                     clust_threshold,
                                                     recalculate_checked,
                                                     directed=True)
        else:
            filtered_twitter_connections = deepcopy(twitter_connections_json)

    ung = nx.node_link_graph(filtered_twitter_connections)
    _, bidir_edges = get_bidir_edges(ung.to_directed())
    del _
    homophily, heterogeneity, heterogeneity_threshold = get_homophily(ung)
    transitivity = nx.transitivity(ung)

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
               "recalculate_coms_checked": int(recalculate_coms_checked),
               "nodes_number": len(filtered_twitter_connections["nodes"]),
               "edges_number": len(filtered_twitter_connections["links"]),
               "is_filtering":int(filtering),
               "dates":dates,
               "dates_dumped":json.dumps(list(dates)),
               "date_index":date_index,
               "current_date":dates[date_index],
               "foci_checked":int(foci_checked),
               "clust_threshold":clust_threshold,
               "bidir":int(bidir),
               "transitivity_value":transitivity,
               "homophily_value": homophily,
               "heterogeneity":heterogeneity,
               "heterogeneity_threshold":heterogeneity_threshold,
               "bidir_edges":bidir_edges,
               "bidir_ratio":bidir_edges*100/len(filtered_twitter_connections["links"]),
               "recalculate_SCCs_checked":int(recalculate_SCCs_checked)}
    if avgs:
        context.update(avgs)

    return render(request, "twitter_connections.html", context)


@lru_cache(maxsize=None)
def filter_by(g, degree, btw, pagerank, closeness, eigenv, clust_threshold,recalculate_node_metrics, directed=True):
    metrics = {"degree":degree, "betweenness":btw, "pagerank":pagerank,
               "closeness_centrality":closeness, "eigenvector_centrality":eigenv,
               "clustering_coefficient":clust_threshold}
    c = g.copy()
    for node in g.nodes():
        invalid = False
        for metric in metrics:
            threshold = metrics[metric]
            try:
                if g.nodes[node][metric] < threshold:
                    invalid = True
                    break
            except KeyError:
                continue
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
