import sys, os
sys.path.insert(0, os.path.abspath(".."))

import argparse
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams.update({'font.size': 16})
import seaborn as sns

import random
from sklearn.manifold import TSNE
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA

from common import data
from common import models
from common import utils
from subgraph_matching.config import parse_encoder

# Now we load the model and a dataset to analyze embeddings on, here ENZYMES.

from subgraph_matching.train import make_data_source

parser = argparse.ArgumentParser()

utils.parse_optimizer(parser)
parse_encoder(parser)
args = parser.parse_args("")
args.model_path = os.path.join("..", args.model_path)

print("Using dataset {}".format(args.dataset))
model = models.OrderEmbedder(1, args.hidden_dim, args)
model.to(utils.get_device())
model.eval()
model.load_state_dict(torch.load(args.model_path,
    map_location=utils.get_device()))

train, test, task = data.load_dataset("wn18")

from collections import Counter

done = False
train_accs = []
while not done:
    data_source = make_data_source(args)
    loaders = data_source.gen_data_loaders(args.eval_interval *
                                           args.batch_size, args.batch_size, train=True)
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):

        pos_a, pos_b, neg_a, neg_b, _ = data_source.gen_batch(batch_target,
                                                              batch_neg_target, batch_neg_query, True, one_small=True)

        pos_a_g, pos_b_g, neg_a_g, neg_b_g, pos_a_anchors, pos_b_anchors, neg_a_anchors, neg_b_anchors = _

        emb_pos_a, emb_pos_b = model.emb_model(pos_a), model.emb_model(pos_b)
        emb_neg_a, emb_neg_b = model.emb_model(neg_a), model.emb_model(neg_b)

        emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
        emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
        labels = torch.tensor([1] * pos_a.num_graphs + [0] * neg_a.num_graphs).to(
            utils.get_device())
        intersect_embs = None
        pred = model(emb_as, emb_bs)
        loss = model.criterion(pred, intersect_embs, labels)

        if args.method_type == "order":
            with torch.no_grad():
                pred = model.predict(pred)
            model.clf_model.zero_grad()
            pred = model.clf_model(pred.unsqueeze(1))
            criterion = nn.NLLLoss()
            clf_loss = criterion(pred, labels)

        pred = pred.argmax(dim=-1)
        acc = torch.mean((pred == labels).type(torch.float))
        train_loss = loss.item()
        train_acc = acc.item()

        print(train_acc)

        train_accs.append(train_acc)

        failed_0 = np.argwhere(((pred != labels) & (labels == 0)).detach().cpu()).reshape(-1).numpy()
        success_0 = np.argwhere(((pred == labels) & (labels == 0)).detach().cpu()).reshape(-1).numpy()

        a_g = pos_a_g + neg_a_g
        b_g = pos_b_g + neg_b_g

        a_g_failed = [a_g[i] for i in failed_0]
        b_g_failed = [b_g[i] for i in failed_0]
        lab_failed = [labels[i].item() for i in failed_0]

        a_g_suc = [a_g[i] for i in success_0]
        b_g_suc = [b_g[i] for i in success_0]
        lab_suc = [labels[i].item() for i in success_0]


        break
        if len(train_accs) > 1:
            break
    break
    if len(train_accs) > 1:
        break
c = -1
for i in range(len(pos_a_g)):
    c += 1
    a = pos_a_g[i]
    b = pos_b_g[i]

    colors = [a[u][v]['edge_type'] for u, v in a.edges]
    lay_a = nx.spring_layout(a)
    if 0 in a.nodes:
        nx.draw(a, edge_color=colors, node_color=[1] + [0] * (len(a) - 1), pos=lay_a)
    else:
        nx.draw(a, edge_color=colors, pos=lay_a)
    plt.text(0,0, F'failed negative a {c}')
    plt.show()


    colors = [b[u][v]['edge_type'] for u, v in b.edges]
    if 0 in b.nodes:
        nx.draw(b, edge_color=colors, node_color=[1] + [0] * (len(b) - 1), pos=lay_a)
    else:
        nx.draw(b, edge_color=colors, pos=lay_a)
    plt.text(0,0, F'failed negative b {c}')
    plt.show()

    print('-' * 50)
c = -1
for i in range(len(lab_failed)):
    c += 1
    a = a_g_failed[i]
    b = b_g_failed[i]

    colors = [b[u][v]['edge_type'] for u, v in b.edges]

    if 0 in b.nodes:
        nx.draw(b, edge_color=colors)
    else:
        nx.draw(b, edge_color=colors)
    plt.text(0,0, F'success negative b {c}')
    plt.show()

    colors = [a[u][v]['edge_type'] for u, v in a.edges]

    if 0 in a.nodes:
        nx.draw(a, edge_color=colors)
    else:
        nx.draw(a, edge_color=colors)

    plt.text(0,0, F'success negative a {c}')
    plt.show()

