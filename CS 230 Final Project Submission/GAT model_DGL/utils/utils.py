import datetime
import dgl
import numpy as np
import random
import torch
import torch.nn.functional as F

from dgl import model_zoo
from dgl.data.chem import smiles_to_bigraph, one_hot_encoding, RandomSplitter
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt 


def set_random_seed(seed=0):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self, epoch_):
        """Compute roc-auc score for each task.

        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0) 
        y_pred = torch.sigmoid(y_pred) 
        #print('y_pred', y_pred.shape) 
        #print(y_pred) 
        predictions = np.round(y_pred) 
        #print(predictions) 
        #print('y_true', y_true.shape) 
        #print(y_true) 
        num_correct = np.where(predictions==y_true)[0] 
        #print('num_correct', len(num_correct)/(783*12)) 
        true_task1 = y_true[:, 0] 
        pred_task1 = predictions[:, 0] 
        #print('task1', len(true_task1)) 
        #print(true_task1) 
        #print('task1', len(pred_task1)) 
        #print(pred_task1) 

        # Creating the histogram for plotting the # of tasks correctly 
        # predicted for each molecule 

        indicators = np.zeros((783, 12))  
        for i in range(783): 
            for j in range(12): 
                if(predictions[i, j] == y_true[i, j]): 
                    indicators[i, j] = 1 
                else: 
                    indicators[i, j] = 0 
        #print("indictators", len(indicators))  
        #print(indicators) 
        summed = np.sum(indicators, axis = 1) 
        #print("summed", len(summed)) 
        #print(summed) 
        dict1 = {} 
        dict2 = {} 
        index = -1 
        for element in summed: 
            index += 1 
            key = element  
            if key in dict1:
                dict1[key] += 1
            else:
                dict1[key] = 1 
            if key in dict2: 
                dict2[key].append(index) 
            else: 
                dict2[key] = [index] 
        print(dict1) 
        plt.bar(dict1.keys(), dict1.values(), 1.0, color='g')
        filename = 'hist_' + str(epoch_) + '.png' 
        plt.savefig(filename) 

        # Examples molecules for 3/12, 6/12, 9/12, 12/12 buckets
        list3 = [] 
        list6 = [] 
        list9 = [] 
        list12 = [] 
        examples3 = [] 
        examples6 = [] 
        examples9 = [] 
        examples12 = [] 
        if 3 in dict2: 
            list3 = dict2[3] 
            if(len(list3) >= 5): 
                examples3 = random.sample(list3, 5) 
        if 6 in dict2: 
            list6 = dict2[6] 
            if(len(list6) >= 5): 
                examples6 = random.sample(list6, 5) 
        if 9 in dict2: 
            list9 = dict2[9] 
            if(len(list9) >= 5): 
                examples9 = random.sample(list9, 5) 
        if 12 in dict2: 
            list12 = dict2[12] 
            if(len(list12) >= 5): 
                examples12 = random.sample(list12, 5)         

        # ROC-AUC scores 
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred)) 
        #print(scores) 
        return scores, examples3, examples6, examples9, examples12 

    def compute_metric(self, metric_name, epoch_=0, reduction='mean'): 
        """Compute metric for each task.

        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task

        Returns
        -------
        list of float
            Metric value for each task
        """
        if metric_name == 'roc_auc':
            return self.roc_auc_score(epoch_) 

class EarlyStopping(object):
    """Early stop performing

    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop
        if the metric stops getting improved
    filename : str or None
        Filename for storing the model checkpoint
    """
    def __init__(self, mode='higher', patience=10, filename=None):
        if filename is None:
            dt = datetime.datetime.now()
            filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks

def load_dataset_for_classification(args):
    """Load dataset for classification tasks.

    Parameters
    ----------
    args : dict
        Configurations.

    Returns
    -------
    dataset
        The whole dataset.
    train_set
        Subset for training.
    val_set
        Subset for validation.
    test_set
        Subset for test.
    """
    assert args['dataset'] in ['Tox21']
    if args['dataset'] == 'Tox21':
        from dgl.data.chem import Tox21
        dataset = Tox21(smiles_to_bigraph, args['atom_featurizer'])
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])

    return dataset, train_set, val_set, test_set



def load_model(args):
    if args['model'] == 'GAT':
        model = model_zoo.chem.GATClassifier(in_feats=args['in_feats'],
                                             gat_hidden_feats=args['gat_hidden_feats'],
                                             num_heads=args['num_heads'],
                                             classifier_hidden_feats=args['classifier_hidden_feats'],
                                             n_tasks=args['n_tasks'])
    return model
