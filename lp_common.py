import numpy as np
import sklearn.metrics as skm
import random
import math
import pickle
import networkx as nx
from node2vec import Node2Vec
from wikipedia2vec import Wikipedia2Vec
import json
import os


class Funcs:
    @staticmethod
    def rand_int(lower, upper):
        return random.randint(lower, upper)

    @staticmethod
    def shuffle_list(target):
        return random.shuffle(target)

    @staticmethod
    def rand_prob():
        return random.random()


class BiMap:
    def __init__(self):
        self.index = 0
        self.forward = dict()
        self.backward = dict()

    def inverse(self, index):
        return self.backward[int(index)]

    def size(self):
        return len(self.forward)

    def get(self, item):
        if item not in self.forward:
            self.forward[item] = self.index
            self.backward[self.index] = item
            self.index += 1

        return self.forward[item]



class GraphData:
    NO_EDGE = "NO_EDGE"
    HAS_EDGE = "HAS_EDGE"
    VIRTUAL_EDGE = "VIRTUAL_EDGE"

    def __init__(self):
        # nodes: index of entities
        self.graph = dict()
        self.node_map = BiMap()

        # relations of edges in a graph. useful for knowledge graph
        self.relations = BiMap()

        # store the global info of a graph, such as ["features"], ["events"] and ["trajectories"]
        self.states = dict()


class Context:
    """
    all configurable fields of the project
    """
    # 1. loader
    multi_class = False  # True / False
    twitter_feat_dim = 500  # the real dimension may be a little bit more than this value
    embedding_method = 0  # 0: original embeddings / features, 1: node2vec
    feature_concat_embedding = False  # True / False, new embeddings with original embeddings / features together
    neg_pos_ratio = 1
    use_shuffle = False
    max_node = -1

    # 2. generator
    tr_ge_divide_ratio = 0.5

    # x. common
    work_dir = "./"
    dataset_dir = work_dir + "../datasets/"


class Profile(Context):
    C = Context
    profile_a = "dblp"
    profile_b = "wiki1k"
    profile_c = "wiki5k"
    profile_d = "wiki10k"
    profile_e = "ppi"
    profile_f = "twitter"
    profile_g = "blogcatalog"
    profile = os.environ["profile"]

    if profile == profile_a:
        dataset = "dblp"
    if profile == profile_b:
        dataset = "wikidata1k"
    if profile == profile_c:
        dataset = "wikidata5k"
    if profile == profile_d:
        dataset = "wikidata10k"
    if profile == profile_e:
        dataset = "ppi"
    if profile == profile_f:
        dataset = "twitter"
    if profile == profile_g:
        dataset = "blogcatalog"


P = Profile


class LPLoader:
    def __init__(self):
        self.graph_data = GraphData()
        self.graph_data.relations.get(GraphData.NO_EDGE)
        if not P.multi_class:
            print("add type HAS_EDGE to relations.")
            self.graph_data.relations.get(GraphData.HAS_EDGE)

    def load_data(self):
        print(P.dataset)
        if P.dataset == "dblp":
            graph = self.attributed_graph('dblp')
        if P.dataset == "wikidata1k":
            graph = self.wiki(1000)
        if P.dataset == "wikidata5k":
            graph = self.wiki(5000)
        if P.dataset == "wikidata10k":
            graph = self.wiki(10000)
        if P.dataset == "ppi":
            graph = self.ppi(3000)
        if P.dataset == "twitter":
            graph = self.twitter(100)
        if P.dataset == "blogcatalog":
            graph = self.attributed_graph('blogcatalog')

        # add embedding (optional)
        if P.embedding_method != 0:
            self.add_embedding(graph)

        # edge (from, to, relation)
        train_edges, train_edges_false, test_edges, test_edges_false = self.tvt_split(graph)
        if not P.multi_class:
            all_edges = train_edges + train_edges_false + test_edges + test_edges_false
        else:
            all_edges = train_edges + test_edges
            print(f"relations: {len(self.graph_data.relations.forward)}")

        # edge / node role exchange
        edge2relation = dict()
        get_prev = dict()
        get_next = dict()
        for ind, [f, t, r] in enumerate(all_edges):
            edge2relation[(f, t)] = r

            if f not in get_next:
                get_next[f] = set()
            get_next[f].add(t)
            
            if t not in get_prev:
                get_prev[t] = set()
            get_prev[t].add(f)

        # convert to target format
        used_edge = set()
        tmp = []
        for f, t, r in all_edges:
            current_edge = (f, t)
            used_edge.add(current_edge)
            if f in get_prev:
                for ff in get_prev[f]:
                    prev_edge = (ff, f)
                    used_edge.add(prev_edge)
                    tmp.append([prev_edge, current_edge])

            if t in get_next:
                for tt in get_next[t]:
                    next_edge = (t, tt)
                    used_edge.add(next_edge)
                    tmp.append([current_edge, next_edge])

        labels = []
        edge2id = dict()
        for ind, edge in enumerate(used_edge):
            edge2id[edge] = ind
            labels.append(edge2relation[edge])

        dgl_dict = dict()
        dgl_dict[("edge", "node", "edge")] = []
        for a, b in tmp:
            dgl_dict[("edge", "node", "edge")].append([
                edge2id[a],
                edge2id[b]
            ])
        
        return dgl_dict, labels

    def tvt_split(self, graph):
        # feeder
        positive_sample = []
        negative_sample = []

        for node_id in graph:
            for edge in graph[node_id]["edges"]:
                if edge in graph:
                    positive_sample.append((node_id, edge, graph[node_id]["edges"][edge]))

        if not P.multi_class:
            node_num = len(graph)
            target_neg_num = math.ceil(len(positive_sample) * P.neg_pos_ratio)
            if target_neg_num < 1:
                target_neg_num = 1
            while len(negative_sample) < target_neg_num:
                rand_from_node = Funcs.rand_int(0, node_num - 1)
                rand_to_node = Funcs.rand_int(0, node_num - 1)
                from_node = list(graph.keys())[rand_from_node]
                to_node = list(graph.keys())[rand_to_node]
                if (from_node, to_node) not in positive_sample:
                    negative_sample.append((from_node, to_node, 0))

        # shuffle train and test
        if P.use_shuffle:
            Funcs.shuffle_list(positive_sample)
            Funcs.shuffle_list(negative_sample)

        train_pos = positive_sample[:int(len(positive_sample) * P.tr_ge_divide_ratio)]
        test_pos = positive_sample[int(len(positive_sample) * P.tr_ge_divide_ratio):]

        train_neg = negative_sample[:int(len(negative_sample) * P.tr_ge_divide_ratio)]
        test_neg = negative_sample[int(len(negative_sample) * P.tr_ge_divide_ratio):]

        return train_pos, train_neg, test_pos, test_neg

    def add_embedding(self, graph, undirected=False):
        # Make graph
        nodes = list(graph)
        edges = []
        embeddings = None
        for node in graph:
            for edge in graph[node]["edges"]:
                if edge in graph:
                    edges.append((int(node), int(edge)))
        if P.embedding_method == 1:  # node2vec
            nx_graph = nx.DiGraph()
            nx_graph.add_nodes_from(nodes)
            nx_graph.add_edges_from(edges)
            # undirected graph
            if undirected:
                nx_graph.to_undirected()
            node2vec = Node2Vec(
                nx_graph, dimensions=256, walk_length=30, num_walks=200, workers=4
            )
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            embeddings = model.wv

        for e in graph:
            if P.feature_concat_embedding:
                if P.embedding_method == 1:  # node2vec
                    graph[e]["feature"] = np.concatenate((
                        graph[e]["feature"],
                        np.array(list(embeddings[str(e)]), dtype=float)
                    ), axis=0)
            else:
                if P.embedding_method == 1:  # node2vec
                    graph[e]["feature"] = np.array(list(embeddings[str(e)]), dtype=float)

    def wiki(self, num):
        wiki2vec = Wikipedia2Vec.load(P.dataset_dir + "wikidata/enwiki_20180420_win10_500d.pkl")
        with open(P.dataset_dir + "wikidata/wikidata-20150921-250k.json", "r") as raw_data:
            entity_num = 0
            edge_num = 0
            for line in raw_data:
                # constraint of max entity
                if entity_num >= num:
                    break

                # filter
                if line.startswith("["):
                    continue
                if line.startswith("]"):
                    continue

                # convert to json format
                json_raw = json.loads(line.strip().strip(","))

                # filter properties
                if str(json_raw["id"]).startswith("P"):
                    continue

                # entity_dict is { 'edges' : {}, 'feature' : {}}
                entity_dict = dict()

                # edges is like { 1 : 3, 2 : 9 } or { 1 : 0, 2 : 1 }
                edges = dict()
                if "claims" not in json_raw:
                    continue
                for claim in json_raw["claims"]:
                    for link in json_raw["claims"][claim]:
                        if str(link["type"]).startswith("statement"):
                            if "datavalue" not in link["mainsnak"]:
                                continue
                            if str(link["mainsnak"]["datavalue"]["type"]).startswith("wikibase-entityid"):
                                to_node_ind = int(link["mainsnak"]["datavalue"]["value"]["numeric-id"])
                                if P.multi_class:
                                    edges[to_node_ind] = self.graph_data.relations.get(int(claim[1:]))
                                else:
                                    edges[to_node_ind] = self.graph_data.relations.get(GraphData.HAS_EDGE)
                entity_dict["edges"] = edges
                edge_num += len(edges)

                # feature search
                try:   
                    label = str(json_raw["labels"]["en"]["value"]).title()
                    label = label.replace("The", "the")
                    label = label.replace("And", "and")
                    label = label.replace("Of", "of")
                    # shape of entity_dict["feature"] is (500, )
                    entity_dict["feature"] = np.array(wiki2vec.get_entity_vector(label).tolist(), dtype=float)
                except KeyError:
                    # some entity feature can't be found or has not english label
                    continue

                # add entity_dic to nodes
                node_ind = self.graph_data.node_map.get(int(json_raw["id"][1:]))
                self.graph_data.graph[node_ind] = entity_dict

                entity_num += 1

            # map the index
            map = dict()
            new_graph = dict()
            for i, n in enumerate(self.graph_data.graph):
                map[n] = i
            for n in self.graph_data.graph:
                new_graph[map[n]] = dict()
                new_graph[map[n]]["feature"] = self.graph_data.graph[n]["feature"]
                new_graph[map[n]]["edges"] = dict()
                for e in self.graph_data.graph[n]["edges"]:
                    if e in map:
                        new_graph[map[n]]["edges"][map[e]] = self.graph_data.graph[n]["edges"][e]
            
            return new_graph

    def twitter(self, num=100):
        feature_dicts = dict()
        edge_dicts = dict()
        entity_num = 0
        processed_egonet = []
        for file in os.listdir(P.dataset_dir + "twitter/"):
            if entity_num >= num:
                break

            node = self.graph_data.node_map.get(int(str(file).split(".")[0]))
            if node not in processed_egonet:
                processed_egonet.append(node)
                if node not in feature_dicts:
                    feature_dicts[node] = dict()
                    entity_num += 1
                    edge_dicts[node] = dict()
                keys = list()
                with open(P.dataset_dir + "twitter/" + str(self.graph_data.node_map.inverse(node)) + ".featnames", "r") as f:
                    for line in f:
                        keys.append(line.rstrip().upper().split(" ")[1])
                values = list()
                with open(P.dataset_dir + "twitter/" + str(self.graph_data.node_map.inverse(node)) + ".egofeat", "r") as f:
                    for line in f:
                        values = line.split(" ")
                for key, value in zip(keys, values):
                    feature_dicts[node][key] = int(value)
                with open(P.dataset_dir + "twitter/" + str(self.graph_data.node_map.inverse(node)) + ".feat", "r") as f:
                    for line in f:
                        splits = line.split(" ")
                        sub_node = self.graph_data.node_map.get(int(splits[0]))
                        if sub_node not in feature_dicts:
                            feature_dicts[sub_node] = dict()
                            entity_num += 1
                            edge_dicts[sub_node] = dict()
                        if sub_node not in edge_dicts[node]:  # all nodes in the ego network connect to the ego user
                            edge_dicts[node][sub_node] = self.graph_data.relations.get(GraphData.HAS_EDGE)
                        sub_values = splits[1:]
                        for key, value in zip(keys, sub_values):
                            feature_dicts[sub_node][key] = int(value)

                with open(P.dataset_dir + "twitter/" + str(self.graph_data.node_map.inverse(node)) + ".edges", "r") as f:
                    for line in f:
                        splits = line.split(" ")
                        from_node = self.graph_data.node_map.get(int(splits[0]))
                        to_node = self.graph_data.node_map.get(int(splits[1]))
                        if to_node not in edge_dicts[from_node]:
                            edge_dicts[from_node][to_node] = self.graph_data.relations.get(GraphData.HAS_EDGE)

        # get all features appeared in feature_dicts
        all_feature = dict()
        for node in feature_dicts:
            for key in feature_dicts[node]:
                if key not in all_feature:
                    if feature_dicts[node][key] > 0:
                        all_feature[key] = 1
                else:
                    all_feature[key] += 1

        # select the most common features (top x)
        top_feature = dict()
        top_feature_value = list(all_feature.values())
        top_feature_value.sort(reverse=True)

        # debug: see the dimension
        # print(top_feature_value)  # see the top feature value to judge the dim needed
        # exit(0)

        if P.dataset == "twitter":
            top_feature_value = top_feature_value[:P.twitter_feat_dim]
        for feat in all_feature:
            if all_feature[feat] in top_feature_value:
                top_feature[feat] = True
        print("feature dimension is {}".format(len(top_feature)))

        # generate feature vec for all nodes
        all_feature_dicts = dict()
        for node in feature_dicts:
            feature_vec = list()
            for key in top_feature:
                if key not in feature_dicts[node]:
                    feature_vec.append(0)
                elif feature_dicts[node][key] == 0:
                    feature_vec.append(0)
                elif feature_dicts[node][key] == 1:
                    feature_vec.append(1)
            all_feature_dicts[node] = feature_vec

        # make data
        for node in feature_dicts:
            self.graph_data.graph[node] = dict()
            # edges: edges is like { 1 : 0, 2 : 1 }
            self.graph_data.graph[node]["edges"] = edge_dicts[node]
            self.graph_data.graph[node]["feature"] = np.array(all_feature_dicts[node], dtype=float)

        print("use {} ego nets".format(len(processed_egonet)))

        return self.graph_data.graph

    def ppi(self, num=3000):
        with open(P.dataset_dir + "ppi/" + "ppi-class_map.json", "r") as class_data:
            protein_class = json.load(class_data)
        protein_feats = np.load(P.dataset_dir + "ppi/" + "ppi-feats.npy")
        with open(P.dataset_dir + "ppi/" + "ppi-G.json", "r") as graph_data:
            ppi_graph = json.load(graph_data)

        entity_num = 0
        edge_num = 0
        # entity is { 'edges' : {}, 'feature' : {}}
        for i in range(len(ppi_graph["nodes"])):
            if Funcs.rand_prob() > 0.1:
                continue
            # constraint of max entity
            if entity_num >= num:
                break

            node_id = self.graph_data.node_map.get(ppi_graph["nodes"][i]["id"])
            if node_id not in self.graph_data.graph:
                self.graph_data.graph[node_id] = dict()
                # edges: edges is like { 1 : 0, 2 : 1 }
                self.graph_data.graph[node_id]["edges"] = dict()
                # feature: feature is like {0.1, 0.5, 0.8}
                self.graph_data.graph[node_id]["feature"] = np.array(
                    list(protein_class[str(self.graph_data.node_map.inverse(node_id))]) +
                    list(protein_feats[self.graph_data.node_map.inverse(node_id)]), dtype=float)
                entity_num = entity_num + 1

        for link in ppi_graph["links"]:
            # filtered edges with to_node not in self.graph_data.nodes
            from_node_ind = self.graph_data.node_map.get(link["source"])
            to_node_ind = self.graph_data.node_map.get(link["target"])
            if from_node_ind in self.graph_data.graph and to_node_ind in self.graph_data.graph:
                if to_node_ind not in self.graph_data.graph[from_node_ind]["edges"]:
                    self.graph_data.graph[from_node_ind]["edges"][to_node_ind] = self.graph_data.relations.get(GraphData.HAS_EDGE)
                    edge_num += 1

        return self.graph_data.graph

    def attributed_graph(self, name, redownload=False):
        if redownload:
            import torch_geometric.transforms as T
            from torch_geometric.datasets import AttributedGraphDataset, CitationFull
            dataset = AttributedGraphDataset(P.dataset_dir + str(name).lower(), str(name).lower(), transform=T.NormalizeFeatures())
            data = dataset[0]
            x = data.x.cpu().detach().numpy()  # (5196, 8189)
            edge_index = data.edge_index.cpu().detach().numpy()  # (2, 343486)
            y = data.y.cpu().detach().numpy()  # (5196,), 6 classes

            with open(P.dataset_dir + f"{str(name).lower()}/{str(name).lower()}.pkl", 'wb') as file:
                pickle.dump([x, edge_index, y], file)

        with open(P.dataset_dir + f"{str(name).lower()}/{str(name).lower()}.pkl", 'rb') as file:
            x, edge_index, y = pickle.load(file)

        ori_x = len(x)
        if P.max_node > 0:
            x = x[:P.max_node]
            y = y[:P.max_node]
            from_node = []
            to_node = []
            for ind in range(len(edge_index[0])):
                if edge_index[0][ind] >= P.max_node or edge_index[1][ind] >= P.max_node:
                    continue
                from_node.append(edge_index[0][ind])
                to_node.append(edge_index[1][ind])
            edge_index = np.array([from_node, to_node], dtype=np.int64)
        print(f'{len(x)} / {ori_x}')

        graph = dict()
        for i in range(x.shape[0]):
            graph[i] = dict()
            graph[i]['edges'] = dict()
            graph[i]['feature'] = x[i]
            graph[i]["label"] = y[i]
        for i in range(edge_index.shape[1]):
            from_node = edge_index[0][i]
            to_node = edge_index[1][i]
            graph[from_node]['edges'][to_node] = self.graph_data.relations.get(GraphData.HAS_EDGE)

        return graph


class LPEval():
    @staticmethod
    def eval(predicted, ground_truth, score, multi_class=P.multi_class):
        if len(predicted) == 0:
            print("predicted value is empty.")
            return

        if multi_class:
            # accuracy
            accuracy = skm.accuracy_score(ground_truth, predicted)

            labels = set()
            for e in ground_truth:
                labels.add(e)

            # Micro-F1
            micro_f1 = skm.f1_score(ground_truth, predicted, labels=list(labels), average="micro")

            # Macro-F1
            macro_f1 = skm.f1_score(ground_truth, predicted, labels=list(labels), average="macro")

            print("Acc: {:.4f} Micro-F1: {:.4f} Macro-F1: {:.4f}".format(accuracy, micro_f1, macro_f1))
        else:
            # auc
            auc = skm.roc_auc_score(ground_truth, score)

            # accuracy
            accuracy = skm.accuracy_score(ground_truth, predicted)

            # recall
            recall = skm.recall_score(ground_truth, predicted)

            # precision
            precision = skm.precision_score(ground_truth, predicted)

            # F1
            f1 = skm.f1_score(ground_truth, predicted)

            # AUPR
            pr, re, _ = skm.precision_recall_curve(ground_truth, score)
            aupr = skm.auc(re, pr)

            # AP
            ap = skm.average_precision_score(ground_truth, score)

            print("Acc: {:.4f} AUC: {:.4f} Pr: {:.4f} Re: {:.4f} F1: {:.4f} AUPR: {:.4f} AP: {:.4f}".format(accuracy, auc, precision, recall, f1, aupr, ap))
