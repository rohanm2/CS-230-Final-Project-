"""GNN layers for updating atom representations"""
import torch.nn as nn
import torch.nn.functional as F

from gatconv import GATConv 

class GATLayer(nn.Module): 
    """Single layer GAT for updating node features

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features for each attention head
    num_heads : int
        Number of attention heads
    feat_drop : float
        Dropout applied to the input features
    attn_drop : float
        Dropout applied to attention values of edges
    alpha : float
        Hyperparameter in LeakyReLU, slope for negative values. Default to be 0.2
    residual : bool
        Whether to perform skip connection, default to be False
    agg_mode : str
        The way to aggregate multi-head attention results, can be either
        'flatten' for concatenating all head results or 'mean' for averaging
        all head results
    activation : activation function or None
        Activation function applied to aggregated multi-head results, default to be None.
    """
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_mode='flatten', activation=None):
        super(GATLayer, self).__init__()
        self.gnn = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                           feat_drop=feat_drop, attn_drop=attn_drop,
                           negative_slope=alpha, residual=residual)
        assert agg_mode in ['flatten', 'mean']
        self.agg_mode = agg_mode
        self.activation = activation

    def forward(self, bg, feats):
        """Update atom representations

        Parameters
        ----------
        bg : DGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size. If self.agg_mode == 'flatten', this would
              be out_feats * num_heads, else it would be just out_feats.
        """
        new_feats = self.gnn(bg, feats)
        if self.agg_mode == 'flatten':
            new_feats = new_feats.flatten(1)
        else:
            new_feats = new_feats.mean(1)

        if self.activation is not None:
            new_feats = self.activation(new_feats)

        return new_feats
