import numpy as np
from heapq import heappush, nlargest
import warnings
import unicodedata, re

try:
    from sklearn.metrics.pairwise import cosine_similarity

    usesklearn = True
except ImportError:
    usesklearn = False
try:
    import networkx as nx
except ImportError:
    warnings.warn("Could not import networkx. Constructing cluster network is not available")
    pass

from itertools import product

from TurkishStemmer import TurkishStemmer
from textblob import TextBlob
from textblob.exceptions import NotTranslated

import gensim, pickle


class ComparableVector(object):
    def __init__(self, score, vector):
        self.score = score
        self.vector = vector

    def __gt__(self, other):
        return self.score>other.score

class Cluster:
    def __init__(self, samples=5, keep_authors_status=False):
        self.cluster_vector = None
        self.vectors_sim = None  # [(similarity, np.zeros(vector_size))]
        self.vectors = list()  # [list of cluster vectors]
        self.authors = dict()
        self.keep_authors_status = keep_authors_status
        self.samples = [None for i in range(samples)]
        self._nonesamples = samples - 1

    def root_similarity(self, v1):
        """
            similarity_to_cluster_vector
        :param v1: np 1-D array
        :return: similarity to the cluster vector
        """
        return self.cos_sim(v1, self.cluster_vector)

    def get_top_n(self, n=5):
        """
        :param n: number of most similar vectors
        :return: most similar n vectors to that cluster
        """
        ## TODO return items
        if n > len(self.vectors):
            warnings.warn("n is bigger than the number of vectors in that cluster")
        return nlargest(min(n, len(self.vectors)), self.vectors_sim)

    @staticmethod
    def cos_sim(v1, v2):
        if usesklearn:
            return np.float64(cosine_similarity(np.atleast_2d(v1), np.atleast_2d(v2)))
        else:
            return np.float64(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def update_cluster_vector(self):
        self.vectors_sim = list()
        self.cluster_vector = np.mean(np.array(self.vectors), axis=0)
        for vector in self.vectors:
            heappush(self.vectors_sim, ComparableVector(self.root_similarity(vector), vector))
            # self.vectors_sim.append((self.root_similarity(vector), vector))

    #         assert self.cluster_vector is not None

    def addtext(self, text, author):
        if self.keep_authors_status and author:
            self.authors.setdefault(author, list())
            self.authors[author].append(text)
        else:
            self.authors.setdefault(author, 0)
            self.authors[author] += 1

        if self._nonesamples >= 0:
            self.samples[self._nonesamples] = text
            self._nonesamples -= 1
        elif np.random.choice([True, False]):
            self.samples[np.random.randint(0, len(self.samples))] = text

    def add(self, vector, text, author):
        #         try:
        self.vectors.append(vector)
        self.addtext(text, author)
        self.update_cluster_vector()

    #         except:
    #             warnings.warn("An error occured during updating the cluster metrics. Last changes reveresed.")
    #             return True

    def __hash__(self):
        h = hash(str(self.cluster_vector))
        for i in self.vectors[:min(len(self.vectors), 3)]:
            h *= hash(str(i))
        return h


class AdaptiveOnlineClustering:

    def __init__(self, en_w2v, tr_w2v, similarity_threshold=0.75, cluster_nsamples=5, vector_size=300):
        self.en_w2v = en_w2v
        self.tr_w2v = tr_w2v
        self.vector_size = vector_size
        self.turkish_stemmer = TurkishStemmer()
        self.similarity_threshold = similarity_threshold
        self.cluster_nsamples = cluster_nsamples
        self.clusters = dict()

    def add(self, text, language="en", author=None, translate=True, stem=False):
        vec = self.vectorize(text, language, translate=translate, stem=stem)
        if vec is None:
            warnings.warn("Invalid text. Document skipped")
        else:
            self._add(vec, text, author)

    def _add(self, vector, text, author):
        highest_similarity = 0
        assigned_cluster = None
        for cluster in self.clusters:
            sim = self.clusters[cluster].root_similarity(vector)
            if sim > highest_similarity:
                highest_similarity = sim
                assigned_cluster = cluster
        if highest_similarity >= self.similarity_threshold:
            self.clusters[assigned_cluster].add(vector, text, author)
        else:
            new_cluster = Cluster(self.cluster_nsamples)
            added = new_cluster.add(vector, text, author)
            self.clusters[len(self.clusters)] = new_cluster
        self._update_clusters()

    def _update_clusters(self):
        for cluster in self.clusters:
            if len(self.clusters[cluster].vectors) < 1:
                del self.clusters[cluster]

    def vectorize(self, text, language, translate=True, stem=False):
        blob = self.clean(text, language, translate=translate, stem=stem)
        if not blob:
            return
        vector = np.zeros(self.vector_size)
        if len(blob.words) < 1:
            return None

        for word in blob.words:
            try:
                if language == "en" or translate:
                    vector += self.en_w2v[word]
                else:
                    vector += self.tr_w2v[word]
            except KeyError:
                continue
        vector /= len(blob.words)
        return vector

    def clean(self, text, language="en", translate=True, stem=False):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').lower().decode("ascii")
        # if language == "tr":
        #     if stem:
        #         text= ' '.join([self.turkish_stemmer.stem(w) for w in text.split()])
        blob = TextBlob(text)
        if translate and language != "en":
            try:
                blob = blob.translate(to="en")
            except NotTranslated:
                return
        text = str(blob)
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r'[0-9]', '#', text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ", text)
        text = re.sub(r"\+", " ", text)
        text = re.sub(r"\-", " ", text)
        text = re.sub(r"\=", " ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r":", " ", text)
        text = re.sub(r"e(\s)?-(\s)?mail", "email", text)

        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        return TextBlob(text)

    def get_weight_threshold(self, weights, weight_threshold_std=2):
        std = np.std(weights)
        avg = np.mean(weights)
        return (avg-std*weight_threshold_std, avg+std*weight_threshold_std)

    def get_network_graph(self, weight_threshold=None, directed=False, weight_threshold_std=2, print_prob=0.0):
        '''

        :param weight_threshold:  (number: minw, number: maxw)
        :param directed: Boolean
        :param weight_threshold_std: (number: how many std units can the weight be from the mean)
        :param print_prob: (probability of printing a connection while being added)
        :return:
        '''
        network_dict = dict()  # {(from,to): weight}
        for c in self.clusters.values():
            for f, t in product(c.authors, c.authors):
                if f == t: continue
                if (f, t) in network_dict:
                    network_dict[(f, t)] += 1
                elif directed:
                    network_dict[(f, t)] = 1
                elif (t, f) in network_dict:
                    network_dict[(t, f)] += 1
                else:
                    network_dict[(f, t)] = 1

        if weight_threshold is None:
            weight_threshold = self.get_weight_threshold(list(network_dict.values()), weight_threshold_std)

        minw, maxw = weight_threshold

        g = nx.Graph()
        for (f, t), weight in network_dict.items():
            # if weight < minw or weight>maxw: continue
            if weight < minw: continue
            if np.random.choice(a=[False, True], p=[1 - print_prob, print_prob]):
                print(f, t, weight)
            g.add_edge(f, t, weight=weight)
        return g