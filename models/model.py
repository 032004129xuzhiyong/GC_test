# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月31日
"""
from mytool import tool
from models.layers import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCDB_model(nn.Module):
    def __init__(self, n_view, n_feats, n_class,
                 n_layer, hid_dim, alpha, lamda):
        super().__init__()
        self.n_view = n_view
        self.alpha = alpha
        self.lamda = lamda

        self.layers_0 = nn.ModuleList([
            nn.Linear(n_feats[i], hid_dim) for i in range(n_view)
        ])
        self.layers = nn.ModuleList([
            GCDBLayer(hid_dim, n_view) for _ in range(n_layer)
        ])
        self.output = nn.Linear(hid_dim, n_class)

    def forward(self, x_list_and_adj_list):
        x_list, adj_list = x_list_and_adj_list
        adj0 = torch.stack(adj_list).mean(0)  # [n,n]
        h0s = []
        for i, layer in enumerate(self.layers_0):
            h0s.append(F.dropout(F.relu(layer(x_list[i])), 0.1, training=self.training))
        Z0 = torch.stack(h0s).mean(0)  # [n, hid_dim]
        Z = Z0
        for i, layer in enumerate(self.layers):
            beta = math.log(self.lamda / (i + 1) + 1)
            Z = F.dropout(F.relu(Z), 0.1, training=self.training)
            Z = layer(Z, h0s, adj0, self.alpha, beta)
        Z = F.dropout(F.relu(Z), 0.1, training=self.training)
        Z = self.output(Z)  # [n, n_class]
        return Z

class GC_weight(nn.Module):
    def __init__(self,n_view, n_feats, n_class,
                 n_layer, hid_dim, alpha, lamda, dropout=0.5):
        super().__init__()
        self.n_view = n_view
        self.alpha = alpha
        self.lamda = lamda
        self.dropout = dropout
        
        self.layers_0 = nn.ModuleList([
            nn.Linear(n_feats[i], hid_dim) for i in range(n_view)
        ])
        self.weight = nn.Parameter(torch.ones(n_view)/n_view, requires_grad=True)
        self.layers = nn.ModuleList([
            GCNIILayer(hid_dim, hid_dim) for _ in range(n_layer)
        ])
        self.output = nn.Linear(hid_dim, n_class)

    def forward(self, x_list_and_adj_list):
        x_list, adj_list = x_list_and_adj_list
        avg_adj = torch.stack(adj_list).mean(0)  # [n,n]
        # make every feature map have the same dimension
        h0s = []
        for i, layer in enumerate(self.layers_0):
            h0s.append(F.dropout(F.relu(layer(x_list[i])), 0.1, training=self.training))
        # weight
        weight = F.softmax(self.weight, dim=0)
        x = torch.zeros_like(h0s[0])
        for i, h0 in enumerate(h0s):
            x = x + weight[i] * h0
        _hidden = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(x)
        _hidden.append(x)
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, self.dropout, training=self.training)
            beta = math.log(self.lamda/(i+1)+1)
            x = F.relu(layer(x, avg_adj, _hidden[0], self.alpha, beta))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.output(x)
        return x


class GC_Att(nn.Module):
    def __init__(self,n_view, n_feats, n_class,
                 n_layer, hid_dim, alpha, lamda, dropout=0.5):
        super().__init__()
        self.n_view = n_view
        self.n_layer = n_layer
        self.alpha = alpha
        self.lamda = lamda
        self.dropout = dropout

        self.layers_0 = nn.ModuleList([
            nn.Linear(n_feats[i], hid_dim) for i in range(n_view)
        ])
        self.weight = nn.Parameter(torch.ones(n_view)/n_view, requires_grad=True)
        self.att_layers = nn.ModuleList([
            Attention(hid_dim, hid_dim, hid_dim) for _ in range(n_layer)
        ])
        self.gc_layers = nn.ModuleList([
            GCNIILayer(hid_dim, hid_dim) for _ in range(n_layer)
        ])
        self.output = nn.Linear(hid_dim, n_class)

    def forward(self, x_list_and_adj_list):
        x_list, adj_list = x_list_and_adj_list
        # make every feature map have the same dimension
        h0s = []
        for i, layer in enumerate(self.layers_0):
            h0s.append(F.dropout(F.relu(layer(x_list[i])), 0.1, training=self.training))
        # weight
        weight = F.softmax(self.weight, dim=0)
        Z0 = torch.zeros_like(h0s[0])
        for i, h0 in enumerate(h0s):
            Z0 = Z0 + weight[i] * h0
        Z0 = F.dropout(Z0, self.dropout, training=self.training)
        Z0 = F.relu(Z0)
        x, adj = Z0, torch.stack(adj_list).mean(0)  # [n,n]
        h0_s, adj0_s = torch.stack(h0s).permute(1,0,2), torch.stack(adj_list).permute(1,0,2)
        for i in range(self.n_layer):
            Z0, att_w = self.att_layers[i](x, h0_s)
            adj = (1-self.alpha) * adj + self.alpha * (adj0_s * att_w).sum(1)
            beta = math.log(self.lamda/(i+1)+1)
            x = F.relu(self.gc_layers[i](x, adj, Z0, self.alpha, beta))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.output(x)
        return x


class AttModel(nn.Module):
    def __init__(self,n_view, n_feats, n_class,
                 n_layer, hid_dim, alpha, lamda, dropout=0.5):
        super().__init__()
        self.n_view = n_view
        self.n_layer = n_layer
        self.alpha = alpha
        self.lamda = lamda
        self.dropout = dropout

        self.proj_layers = nn.ModuleList([
            nn.Linear(n_feats[i], hid_dim) for i in range(n_view)
        ])
        self.self_att = nn.MultiheadAttention(hid_dim, num_heads=8)
        self.agg_att = AggAttention(hid_dim, att_channel=hid_dim)
        self.cross_att_layers = nn.ModuleList([
            CrossAttention(hid_dim, hid_dim, att_channel=hid_dim) for _ in range(n_layer)
        ])
        self.gc_layers = nn.ModuleList([
            GCNIILayer(hid_dim, hid_dim) for _ in range(n_layer)
        ])
        self.output = nn.Linear(hid_dim, n_class)

    def forward(self, x_list_and_adj_list):
        x_list, adj_list = x_list_and_adj_list

        # proj
        h0s = []
        for i, layer in enumerate(self.proj_layers):
            h0s.append(F.dropout(F.relu(layer(x_list[i])), 0.1, training=self.training))
        h0_s = torch.stack(h0s).permute(1,0,2)  # [node, view, hid]
        adj0_s = torch.stack(adj_list).permute(1,0,2)  # [node, view, node]

        # self attention
        # [node, view, hid] [view, node, node]
        attn_o, attn_w = self.self_att(h0_s, h0_s, h0_s)

        # agg attention
        # [node, hid] [1, view, 1]
        agg_z, agg_w = self.agg_att(attn_o)
        agg_adj = (adj0_s * agg_w).sum(1)  # [node, node]

        # forward
        z, adj, z0 = agg_z, agg_adj, h0_s.mean(1)
        for i in range(self.n_layer):
            beta = math.log(self.lamda/(i+1)+1)
            z = F.relu(self.gc_layers[i](z, adj, z0, self.alpha, beta))
            z = F.dropout(z, self.dropout, training=self.training)

            # [node, hid] [1, view, 1]
            z0, agg_w = self.cross_att_layers[i](h0_s, z)
            adj = (1-self.alpha) * adj + self.alpha * (adj0_s * agg_w).sum(1)  # [node, node]
        out = self.output(z)
        return out