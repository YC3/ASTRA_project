import numpy as np
import pandas as pd
import torch
import networkx as nx
import requests, gzip, shutil, tarfile
from pathlib import Path


def download_graph_edge_list(target_dir):
    
    path = Path(target_dir)
    
    url = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
    r = requests.get(url)

    with open(path.joinpath("go.tar.gz"), "wb") as f:
        f.write(r.content)
    f.close()

    f = tarfile.open(path.joinpath('go.tar.gz'))
    f.extractall(path) 
    f.close()
    

def create_graph(filepath, topn=10, nrows=None):
    
    go_ = pd.read_csv(filepath, nrows=nrows)

    go = go_.groupby('target').apply(lambda x: x.nlargest(topn + 1, ['importance'])).reset_index(drop = True)

    gene_list = list(set(go.source.tolist() + go.target.tolist()))
    gene2idx = {g:i for i,g in enumerate(gene_list)}

    G = nx.from_pandas_edgelist(go, source='source',target='target', edge_attr=['importance'], create_using=nx.DiGraph())

    edge_index_ = [(gene2idx[e[0]], gene2idx[e[1]]) for e in G.edges]
    edge_index = torch.tensor(edge_index_, dtype=torch.long).T

    edge_attr = nx.get_edge_attributes(G, 'importance') 
    edge_weight = torch.Tensor(np.array([edge_attr[e] for e in G.edges]))

    return edge_index, edge_weight, gene_list, gene2idx
