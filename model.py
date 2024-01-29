import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet
import config as c
from freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, ReversibleGraphNet, OutputNode, \
    InputNode, Node

WEIGHT_DIR = './weights'
MODEL_DIR = './models'


def nf_head(input_dim, n_coupling_blocks,
            clamp_alpha, fc_internal, dropout):
    nodes = list()
    nodes.append(InputNode(input_dim,
                           name='input'))
    for k in range(n_coupling_blocks):
        init_dim = nodes[-1].out0
        print(f'node {k} init_dim: {init_dim}')
        nodes.append(Node([nodes[-1].out0],
                          permute_layer,
                          {'seed': k},
                          name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0],
                          glow_coupling_layer,
                          {'clamp': clamp_alpha,
                           'F_class': F_fully_connected,
                           'F_args': {'internal_size': fc_internal,
                                      'dropout': dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0],
                            name='output'))
    coder = ReversibleGraphNet(node_list = nodes,
                               ind_in=None,
                               ind_out=None,
                               verbose=False)

    return coder


class DifferNet(nn.Module):
    def __init__(self,
                 n_scales,
                 n_feat,
                 n_coupling_blocks,
                 clamp_alpha,
                 fc_internal,
                 dropout):
        super(DifferNet, self).__init__()
        # ------------------------------------------------------------------------------------------------------------
        # use pretrained alexnet
        self.feature_extractor = alexnet(pretrained=True)
        self.nf = nf_head(input_dim=n_feat,                     # 256 * 3
                          n_coupling_blocks=n_coupling_blocks,  # 8
                          clamp_alpha=clamp_alpha,              # 3
                          fc_internal=fc_internal,              # 2048
                          dropout=dropout)                      # 0.0
        self.n_scales = n_scales # 3
        self.img_size = (448, 448)

    def forward(self, x):
        y_cat = list()

        # ------------------------------------------------------------------------------------------------------------
        # (1) Alexnet
        for s in range(self.n_scales):
            # s = 0, 1, 2
            x_scaled = F.interpolate(x,size=self.img_size[0] // (2 ** s)) if s > 0 else x
            # feature_extractor = alexnet
            feat_s = self.feature_extractor.features(x_scaled)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))
        # ------------------------------------------------------------------------------------------------------------
        # (2) nf head
        y = torch.cat(y_cat, dim=1)
        print(f'y.shape (batch, 256*3) : {y.shape}')
        z = self.nf(y)
        return z


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model
