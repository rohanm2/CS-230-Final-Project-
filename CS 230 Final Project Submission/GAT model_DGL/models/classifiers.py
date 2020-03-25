import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn import GCNLayer, GATLayer  

class GATClassifier(BaseGNNClassifier):
    """GAT based predictor for multitask prediction on molecular graphs.
    We assume each task requires to perform a binary classification.

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    """
    def __init__(self, in_feats, gat_hidden_feats, num_heads,
                 n_tasks, classifier_hidden_feats=128, dropout=0):
        super(GATClassifier, self).__init__(gnn_out_feats=gat_hidden_feats[-1],
                                            n_tasks=n_tasks,
                                            classifier_hidden_feats=classifier_hidden_feats,
                                            dropout=dropout)
        assert len(gat_hidden_feats) == len(num_heads), \
            'Got gat_hidden_feats with length {:d} and num_heads with length {:d}, ' \
            'expect them to be the same.'.format(len(gat_hidden_feats), len(num_heads))
        num_layers = len(num_heads)
        for l in range(num_layers):
            if l > 0:
                in_feats = gat_hidden_feats[l - 1] * num_heads[l - 1]

            if l == num_layers - 1:
                agg_mode = 'mean'
                agg_act = None
            else:
                agg_mode = 'flatten'
                agg_act = F.elu

            self.gnn_layers.append(GATLayer(in_feats, gat_hidden_feats[l], num_heads[l],
                                            feat_drop=dropout, attn_drop=dropout,
                                            agg_mode=agg_mode, activation=agg_act))
