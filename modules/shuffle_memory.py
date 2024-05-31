# Includes all GNN base models used in memory mixing

import torch
from torch_geometric.nn import GCNConv, GATConv, NNConv, GINConv, GATv2Conv
import torch.nn.functional as F
        
class ExpanderGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
    
class ExpanderGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpanderGAT, self).__init__()
        self.conv = GATConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.relu(self.linear(x))

class ExpanderGIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpanderGIN, self).__init__()
        nn = MLP(in_channels, out_channels)
        self.conv = GINConv(nn)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

class ExpanderGATv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpanderGATv2, self).__init__()
        self.conv = GATv2Conv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x