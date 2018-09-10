# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from collections import OrderedDict

try:
    from functools import lru_cache
except ImportError:
    from repoze.lru import lru_cache

from django.http import JsonResponse
from django.shortcuts import render, render_to_response
import pandas as pd
from datetime import datetime
import unicodedata

from copy import deepcopy

import os, json, pickle

import networkx as nx
import snap

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


def get_homophily(nw, metric="lang", decimals=3):
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
    return cross_metric_ratio < heterogeneity_fraction_norm,\
           round(cross_metric_ratio, decimals), round(heterogeneity_fraction_norm, decimals)


def get_bidir_edges(G, bidir):
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

    for i, CnCom in enumerate(components):
        for n in CnCom:
            nodes_sccs[n] = i


    for node in G.nodes():
        m = str(nodes_sccs[node])
        G.nodes[node]["SCC"] = m

    return G


def calculate_communities(G):
    g = networkx_to_snappy(G)
    CmtyV = snap.TCnComV()
    modularity = snap.CommunityGirvanNewman(g, CmtyV)
    nodes_communities = {}  # {node: [community]}
    for i, Cmty in enumerate(CmtyV):
        for NI in Cmty:
            nodes_communities.setdefault(NI, [])
            nodes_communities[NI].append(i + 2)

    return nodes_communities


def label_nodes_communities(G,dates, date_index, bidir, foci_checked,
                            degree_threshold, btw_threshold, pagerank_threshold,
                            closeness_threshold, eigenvector_threshold, clust_threshold,
                            recalculate_coms_checked, save=False):
    filename = "{0}_{1}_{2}.pkl".format(dates[date_index], bidir, foci_checked)
    try:
        nodes_communities = pickle.load(open(root_dir + "/static/communities/"+filename,'rb'))
        print("Reading", filename)

        save = save or not sum([degree_threshold, btw_threshold, pagerank_threshold,
                                closeness_threshold, eigenvector_threshold,
                                clust_threshold])  # The entire network at that date
    except IOError:
        recalculate_coms_checked = True
        save = True

    if recalculate_coms_checked:
        print("CALCULATING COMMUNITIES", len(G.nodes()))
        nodes_communities = calculate_communities(G)
        if save:
            print("SAVING", filename)
            pickle.dump(nodes_communities, open(root_dir + "/static/communities/"+filename, 'wb'))

    for node in G.nodes():
        G.nodes[node]["community"] = nodes_communities[node][0]
    return G

def construct_network(connections, dates, date_index, bidir, foci_checked,
                            degree_threshold, btw_threshold, pagerank_threshold,
                            closeness_threshold, eigenvector_threshold, clust_threshold,
                            recalculate_coms_checked, recalculate_checked):
    G = nx.DiGraph()
    truncate = lambda x: int(str(int(x))[:9])
    for _, row in connections.iterrows():
        f = row["from_user_id"]
        t = row["to_user_id"]
        from_ = truncate(f)
        to = truncate(t)
        if not foci_checked and (from_ in orgs or to in orgs):
            continue
        if from_ in twitter_users.truncated_id and to in twitter_users.truncated_id:
            G.add_edge(from_, to)

    new_G, bidir_edges = get_bidir_edges(G, bidir)

    filtered_g = filter_by(recalculate_metrics(new_G), degree_threshold, btw_threshold, pagerank_threshold, closeness_threshold,
           eigenvector_threshold, clust_threshold, recalculate_checked)

    # augs = ["name", "screen_name", "match_name", "followers_count", "friends_count", "lang"]
    # getting pre-calculated communities >> THIS IS JUST OPTIONAL
    augs = ["name", "screen_name", "match_name", "followers_count", "friends_count", "lang"]

    filtered_g = label_nodes_communities(filtered_g, dates, date_index, bidir, foci_checked,
                            degree_threshold, btw_threshold, pagerank_threshold,
                            closeness_threshold, eigenvector_threshold, clust_threshold,
                            recalculate_coms_checked)

    filtered_g = label_nodes_SCCs(filtered_g)



    for node in filtered_g.nodes():
        user = twitter_users.loc[node]
        for aug in augs:
            filtered_g.nodes[node][aug] = user[aug]

    return filtered_g, bidir_edges



# twitter_connections_json = root_dir + "/static/networks/twitter_users_graph2.json"
# twitter_fb_j<son = root_dir + "/static/networks/twitter_fb.json"


twitter_connections_path = root_dir+ "/static/networks/twitter_users_graph2.json"
twitter_connections_json = json.load(open(twitter_connections_path,"r"))

twitter_fb_path = root_dir + "/static/networks/twitter_fb.json"
twitter_fb_json = json.load(open(twitter_fb_path, "r"))


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


def get_avg_metric(graph, metric, decimals=4):
    result = 0.0
    for node in graph["nodes"]:
        try:
            result += node[metric]
        except KeyError:
            return 0.
    return round(result/len(graph["nodes"]), decimals)


@lru_cache(maxsize=None)
def recalculate_metrics(nxg, centralities=True):

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
        except Exception as e:
            print("ERROR", e)
            pass


    # data = json_graph.node_link_data(nxg)
    # with open(root_dir + "/static/networks/latest_tw_ntw.json", 'w') as f:
    #     json.dump(data, f, indent=4)
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

def get_communities_count(data, group='community'):
    return len(set([i[group] for i in data['nodes']]))


def twitter_connections_raw(request):
    global template_name
    template_name = "twitter_connections_raw.html"
    return twitter_connections(request)


def twitter_connections_reset(request):
    global template_name
    template_name = "twitter_connections.html"
    return twitter_connections(request)


template_name = "twitter_connections.html"
defaults = {"degree_threshold":0, "btw_threshold":0.0, "pagerank_threshold":0.0, "closeness_threshold":0.0,
            "eigenvector_threshold":0.0, "clust_threshold":0.0, "recalculate_checked":0,
            "recalculate_coms_checked":False, "foci_checked":1, "bidir":0, "date_index":0,
            "size_metric":"degree", "filtered_twitter_connections":deepcopy(twitter_connections_json),
            "dates":get_dates()}

size_metrics = ["degree", "in_degree", "out_degree",
                "betweenness", "closeness_centrality", "eigenvector_centrality",
                "pagerank","clustering_coefficient", "followers_count"]

'''Mapping network parameters to the corresponding json-formatted network,
and limiting this custom caching to 100 networks'''
MAX_CACHED_NETWORKS = 100
filtered_networks_dict = OrderedDict()

def hash_parameters(degree_threshold, btw_threshold, pagerank_threshold, closeness_threshold, eigenvector_threshold,
                    clust_threshold, date_index, recalculate_checked, recalculate_coms_checked, bidir, foci_checked):
    return ":".join([
        str(round(float(i), 2)) for i in [degree_threshold, btw_threshold, pagerank_threshold,closeness_threshold,
                                          eigenvector_threshold,clust_threshold,date_index, recalculate_checked,
                                          recalculate_coms_checked, bidir, foci_checked]])

def twitter_connections(request):
    global defaults, size_metrics, template_name, filtered_networks_dict

    default = lambda var_name, def_val: def_val if var_name not in vars() else vars()[var_name]

    degree_threshold = default("degree_threshold", defaults["degree_threshold"])
    btw_threshold = default("btw_threshold", defaults["btw_threshold"])
    pagerank_threshold = default("pagerank_threshold", defaults["pagerank_threshold"])
    closeness_threshold = default("closeness_threshold", defaults["closeness_threshold"])
    eigenvector_threshold = default("eigenvector_threshold", defaults["eigenvector_threshold"])

    recalculate_checked = default("recalculate_checked", defaults["recalculate_checked"])
    foci_checked = default("foci_checked", defaults["foci_checked"])
    dates = default("dates", defaults["dates"])
    date_index = default("date_index", defaults["date_index"])
    clust_threshold = default("clust_threshold", defaults["clust_threshold"])
    bidir = default("bidir", defaults["bidir"])
    recalculate_coms_checked = default("recalculate_coms_checked", defaults["recalculate_coms_checked"])
    try:
        filtered_twitter_connections
    except NameError:
        filtered_twitter_connections = defaults["filtered_twitter_connections"]


    check = {"on":True, False:False}
    filtering = False

    if request.method == "POST":
        filtering = True
        degree_threshold = int(request.POST["text_degree"])
        btw_threshold = float(request.POST["text_betweenness"])
        pagerank_threshold = float(request.POST["text_pagerank"])
        closeness_threshold = float(request.POST["text_closeness_centrality"])
        eigenvector_threshold = float(request.POST["text_eigenvector_centrality"])
        clust_threshold = float(request.POST["text_clustering_coefficient"])
        date_index = int(request.POST["date"])
        recalculate_checked = check[request.POST.get("recalculate_metrics", False)]
        recalculate_coms_checked = check[request.POST.get("recalculate_coms_checked", False)]

        foci_checked = check[request.POST.get("toggleFociCB", False)]
        bidir = check[request.POST.get("toggleBidir", False)]

        do_filter = foci_checked != defaults['foci_checked'] or bidir != defaults['bidir']
        if not do_filter:
            for i in [date_index, degree_threshold, btw_threshold, pagerank_threshold,
                      closeness_threshold, eigenvector_threshold, clust_threshold]:
                if i != 0:
                    do_filter = True
                    break

        if do_filter:
            # if date_index == 0 and foci_checked==DEFAULT_FOCI_CHECKED and bidir==DEFAULT_BIDIR:  # same network
            #     data = deepcopy(twitter_connections_json)
            # else:
            cons = get_connections_by_date(dates[date_index])
            nw, bidir_edges = construct_network(cons, dates, date_index, bidir, foci_checked,
                            degree_threshold, btw_threshold, pagerank_threshold,
                            closeness_threshold, eigenvector_threshold, clust_threshold,
                            recalculate_coms_checked, recalculate_checked)

            filtered_twitter_connections = nx.node_link_data(nw)

        else:
            filtered_twitter_connections = deepcopy(twitter_connections_json)

    ung = nx.node_link_graph(filtered_twitter_connections)
    _, bidir_edges = get_bidir_edges(ung.to_directed(), bidir)
    del _
    homophily, heterogeneity, heterogeneity_threshold = get_homophily(ung)
    transitivity = round(nx.transitivity(ung), 4)

    avgs = None
    if len(filtered_twitter_connections["nodes"])>=1:
        avgs = {"avg_" + metric: get_avg_metric(filtered_twitter_connections, metric) for metric in size_metrics}

    all_colors = json.load(open(root_dir + "/static/communities/colors.json", 'r'))
    colors = [str(c) for c in all_colors[:get_communities_count(filtered_twitter_connections)]]
    del all_colors
    context = {"size_metrics":size_metrics,
               "degree_threshold": degree_threshold,
               "btw_threshold": btw_threshold,
               "pagerank_threshold": pagerank_threshold,
               "closeness_threshold": closeness_threshold,
               "eigenvector_threshold": eigenvector_threshold,
               "clust_threshold": clust_threshold,
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
               "bidir":int(bidir),
               "transitivity_value":transitivity,
               "homophily_value": homophily,
               "heterogeneity":heterogeneity,
               "heterogeneity_threshold":heterogeneity_threshold,
               "bidir_edges":bidir_edges,
               "bidir_ratio":bidir_edges*100/len(filtered_twitter_connections["links"]),
               "colors":colors}
    if avgs:
        context.update(avgs)

    k = hash_parameters(degree_threshold, btw_threshold, pagerank_threshold, closeness_threshold, eigenvector_threshold,
                    clust_threshold, date_index, recalculate_checked, recalculate_coms_checked, bidir, foci_checked)

    filtered_networks_dict[k] = filtered_twitter_connections
    if len(filtered_networks_dict)>MAX_CACHED_NETWORKS:
        del filtered_networks_dict[filtered_networks_dict.keys()[0]]  # remove the oldest network in the cache
    globals()['filtered_twitter_connections'] = filtered_twitter_connections
    return render(request, template_name, context)


@lru_cache(maxsize=None)
def filter_by(g, degree_threshold, btw_threshold, pagerank_threshold, closeness_threshold,
           eigenvector_threshold, clust_threshold, recalculate_checked):

    metrics = {"degree":degree_threshold, "betweenness":btw_threshold, "pagerank":pagerank_threshold,
               "closeness_centrality":closeness_threshold, "eigenvector_centrality":eigenvector_threshold,
               "clustering_coefficient":clust_threshold}
    c = g.copy()
    for node in g.nodes():
        invalid = False
        for metric, threshold in metrics.iteritems():
            if g.nodes[node][metric] < threshold:
                invalid = True
                break
        if invalid:
            c.remove_node(node)
    if recalculate_checked:
        return recalculate_metrics(c)
    return c

# def load_twitter_connections_json(request, degree_threshold):
#     print("DEGREE",degree_threshold)
#
#     return JsonResponse(globals()['filtered_twitter_connections'], safe=False)

def load_twitter_connections_json(request, degree_threshold, btw_threshold, pagerank_threshold,
                                  closeness_threshold, eigenvector_threshold,clust_threshold,
                                  date_index, recalculate_checked, recalculate_coms_checked, bidir, foci_checked):

    k = hash_parameters(degree_threshold, btw_threshold, pagerank_threshold,
                                  closeness_threshold, eigenvector_threshold,clust_threshold,
                                  date_index, recalculate_checked, recalculate_coms_checked, bidir, foci_checked)

    # return JsonResponse(globals()['filtered_twitter_connections'], safe=False)
    return JsonResponse(filtered_networks_dict[k], safe=False)

def twitter_fb(request):
    return JsonResponse(twitter_fb_json, safe=False)

def twitter_fb_connections(request):
    return render_to_response("twitter_fb.html", {})
