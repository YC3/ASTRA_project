import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import SGConv

from typing import List



class MLP(nn.Module):

    def __init__(
        self, 
        sizes: List[int] = None, 
        batch_norm: bool = True
    ):

        super().__init__()
        self.sizes = sizes
        self.batch_norm = batch_norm
        
        layers = []
        for i in range(len(self.sizes) - 1):
            
            in_size = self.sizes[i]
            out_size = self.sizes[i + 1]
            
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size) if self.batch_norm else None,
                nn.ReLU()
            ])

        self.layers = nn.Sequential(*[l for l in layers if l is not None])
        
    def forward(
        self, 
        x: Tensor
    ):
        return self.layers(x)
    
    
class GNN(nn.Module):

    def __init__(
        self, 
        genes: int,
        seq_len: int, # n_cell (this is a fixed value for all genes)
        d_hid: int,
        edge_index: Tensor,
        edge_weight: Tensor,
        gene2idx: dict,
        n_gnn_layers: int,
        device: str = 'cpu',
    ):

        super().__init__()   
        self.genes = genes
        self.n_genes = len(genes)
        self.seq_len = seq_len
        self.d_hid = d_hid if d_hid < 8*(seq_len//32) else 8*(seq_len//32)
        self.d_emb = seq_len + 1
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.gene2idx = gene2idx
        self.device = device

        ### GNN layers
        self.n_gnn_layers = n_gnn_layers
        self.edge_index = edge_index
        self.edge_weight = edge_weight.to(self.device)
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(1, self.n_gnn_layers+1):
            self.gnn_layers.append(SGConv(self.d_emb, self.d_emb, 1))
            
        ### attention layer
        self.attn = nn.MultiheadAttention(self.d_emb, 1, dropout=0.1, bias=True)

        # decoder layer
        self.decoder = MLP([self.d_emb, d_hid*2, d_hid*2, d_hid*4], batch_norm=False)
        self.output = nn.Linear(d_hid*4, self.seq_len)
        

    def forward(
        self, 
        src: Tensor,
        pert_gene: str, # TODO: batch
    ):
       
        pert_idx = self.gene2idx[pert_gene]
        ## pertubation embedding
        pert_emb = torch.zeros((self.n_genes, 1)) # n_gene, 1
        pert_emb[pert_idx, :] = 1 

        ## node feature
        emb = torch.concat((src, pert_emb), dim=1) # n_gene, seq_len+1

        ## augment global perturbation embedding with GNN
        for i, gnn in enumerate(self.gnn_layers):
            emb = gnn(emb, self.edge_index, self.edge_weight)
            if i < self.n_gnn_layers - 1:
                emb = emb.relu()

        ## attention
        attn_output, attn_weights = self.attn(emb, emb, emb)        
        cell_emb = torch.matmul(attn_weights[pert_idx, :], emb).reshape(1, -1)
        
        ## output
        return self.output(self.decoder(cell_emb))
