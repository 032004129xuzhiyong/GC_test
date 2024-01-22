# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月31日
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GCNIILayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # in_channel == out_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = nn.Parameter(torch.FloatTensor(in_channel, out_channel))
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, h0, alpha, beta):
        assert x.size(-1) == h0.size(-1)
        left = (1 - alpha) * torch.mm(adj,x) + alpha * h0
        right = (1-beta) * torch.eye(self.out_channel,device=x.device) + beta * self.weight
        return left @ right


class GCDBLayer(nn.Module):
    def __init__(self, out_channel, n_view):
        super().__init__()
        self.n_view = n_view
        self.weight = nn.Parameter(torch.ones(n_view)/n_view, requires_grad=True)
        self.gcs = nn.ModuleList([
            GCNIILayer(out_channel, out_channel) for _ in range(n_view)
        ])

    def forward(self, Z, h0s, adj0, alpha, beta):
        # GCNII
        z = torch.zeros_like(Z)
        weight = F.softmax(self.weight, dim=0)
        for i, gc in enumerate(self.gcs):
            z = z + weight[i] * gc(Z, adj0, h0s[i], alpha, beta)
        return z


class AggAttention(nn.Module):
    def __init__(self, in_channel, att_channel):
        super().__init__()
        self.attn_proj = nn.Linear(in_channel, att_channel)
        self.act = nn.Tanh()
        self.f1 = nn.Linear(att_channel, 1)

    def forward(self, node_view_hid):
        x = self.attn_proj(node_view_hid)  # [node, view, att_hid]
        x = self.act(x)
        x = self.f1(x)  # [node, view, 1]
        w = x.mean(0).unsqueeze(0)  # [1, view, 1]
        w = F.softmax(w, 1)
        out = (node_view_hid * w).sum(1)  # [node, hid]
        return out, w


class CrossAttention(nn.Module):
    def __init__(self, S_in_channel, U_in_channel, att_channel):
        super().__init__()
        self.h0_s_linear = nn.Linear(S_in_channel, att_channel)
        self.z_linear = nn.Linear(U_in_channel, att_channel)
        self.act = nn.Tanh()
        self.f1 = nn.Linear(att_channel, 1)

    def forward(self, h0_s, z):
        # h0_s: [node, view, hid_dim]
        # z: [node, hid_dim]
        att1 = self.h0_s_linear(h0_s)  # [node, view, att]
        att2 = self.z_linear(z)  # [node, att]
        att = self.act(att1 + att2.unsqueeze(1))  # [node, view, att]
        att = self.f1(att)  # [node, view, 1]
        w = att.mean(0).unsqueeze(0)  # [1, view, 1]
        w = F.softmax(w, 1)
        out = (h0_s * w).sum(1)  # [node, view, hid_dim]
        return out, w


class Attention(nn.Module):
    def __init__(self,S_in_channel, U_in_channel, att_channel):
        super().__init__()
        self.h0_z_linear = nn.Linear(S_in_channel, att_channel)
        self.Z_linear = nn.Linear(U_in_channel, att_channel)
        self.f1 = nn.Linear(att_channel, 1)

    def forward(self, h0_z, Z):
        # Z: [n_node, hid_dim]
        # h0_z: [n_node, n_view, hid_dim]
        att1 = self.Z_linear(Z)
        att2 = self.h0_z_linear(h0_z)
        att = self.f1(F.relu(att1.unsqueeze(1)+att2))  # [n_node, n_view, 1]
        att_w = F.softmax(att,dim=1)
        return (h0_z * att_w).sum(1), att_w  # [n_node, hid_dim]


class GCNII_star_Layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # in_channel == out_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight1 = nn.Parameter(torch.FloatTensor(in_channel, out_channel))
        self.weight2 = nn.Parameter(torch.FloatTensor(in_channel, out_channel))
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, h0, alpha, beta):
        left = (1-alpha) * torch.mm(adj,x) @ \
               ((1-beta)*torch.eye(self.out_channel,device=x.device) + beta * self.weight1)
        right = alpha * h0 @ \
                ((1-beta)*torch.eye(self.out_channel,device=x.device) + beta * self.weight2)
        return left + right


if __name__ == '__main__':
    gc = GCNIILayer(32, 32)
    # gc = GCNII_star_Layer(32,32)
    print(gc(torch.rand(50,32), torch.ones(50,50),torch.rand(50,32), 0.1, 0.5).shape)