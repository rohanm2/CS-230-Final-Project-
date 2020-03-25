import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 

from utils.utils import Meter, EarlyStopping, collate_molgraphs, set_random_seed, \
    load_dataset_for_classification, load_model

from dgl.data.chem import BaseAtomFeaturizer, CanonicalAtomFeaturizer, ConcatFeaturizer, \
    atom_type_one_hot, atom_degree_one_hot, atom_formal_charge, atom_num_radical_electrons, \
    atom_hybridization_one_hot, atom_total_num_H_one_hot, BaseBondFeaturizer

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        atom_feats = bg.ndata.pop(args['atom_data_field'])
        atom_feats, labels, masks = atom_feats.to(args['device']), \
                                    labels.to(args['device']), \
                                    masks.to(args['device'])
        logits = model(bg, atom_feats)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits, labels, masks) 
    # Separated stuff here 
    scores, examples3, examples6, examples9, examples12 = train_meter.compute_metric(args['metric_name']) 
    train_score = np.mean(scores) 
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric_name'], train_score))
    return loss.item() 

from dgl.data.chem import Tox21, smiles_to_bigraph, CanonicalAtomFeaturizer     
from rdkit import Chem
from rdkit.Chem import Draw 
import matplotlib 


def run_an_eval_epoch(args, model, data_loader, epoch, last): 
    model.eval()
    eval_meter = Meter() 
    listOfMolSmiles = [] # 
    with torch.no_grad(): 
        for batch_id, batch_data in enumerate(data_loader): 
            smiles, bg, labels, masks = batch_data 
            for smile in smiles: 
                listOfMolSmiles.append(smile) 
            #print('labels', len(labels)) 
            #print(labels) 
            atom_feats = bg.ndata.pop(args['atom_data_field'])
            atom_feats, labels = atom_feats.to(args['device']), labels.to(args['device'])
            logits = model(bg, atom_feats) 
            #print(len(logits)) 
            #print(logits) 
            eval_meter.update(logits, labels, masks)
    # This is the val_score 
    scores, examples3, examples6, examples9, examples12 = eval_meter.compute_metric(args['metric_name'], epoch_ = epoch) 
    print('scores') 
    print(scores) 

    print('examples3', len(examples3)) 
    print(examples3) 
    print('examples6', len(examples6)) 
    print(examples6) 
    print('examples9', len(examples6)) 
    print(examples9) 
    print('examples12', len(examples6)) 
    print(examples12) 

    print('listOfMolSmiles', len(listOfMolSmiles)) 
    #print(listOfMolSmiles) 

    exampleMols = [] 
    for ex in examples3: 
        exampleMols.append(listOfMolSmiles[ex]) 
    for ex in examples6: 
        exampleMols.append(listOfMolSmiles[ex]) 
    for ex in examples9: 
        exampleMols.append(listOfMolSmiles[ex]) 
    for ex in examples12: 
        exampleMols.append(listOfMolSmiles[ex]) 
    
    flags = [True, True, True, True] 
    if(len(examples3) == 0): 
        flags[0] = False 
    if(len(examples6) == 0): 
        flags[0] = False 
    if(len(examples9) == 0): 
        flags[0] = False 
    if(len(examples12) == 0): 
        flags[0] = False 

    if(last == True): ## 
        print(flags) 
        for i in range(len(exampleMols)): 
            smiles = exampleMols[i] 
            m = Chem.MolFromSmiles(smiles) 
            fileName = 'test_' + str(epoch) + '.' + str(i) + '.png' 
            fig = Draw.MolToFile(m, fileName = fileName) 

    return np.mean(scores) 
    #return np.mean(eval_meter.compute_metric(args['metric_name'])) 

def main(args):
    args['device'] = torch.device("cpu")
    set_random_seed(args['random_seed'])

    dataset, train_set, val_set, test_set = load_dataset_for_classification(args)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)

    args['n_tasks'] = dataset.n_tasks
    model = load_model(args)
    loss_criterion = BCEWithLogitsLoss(pos_weight=dataset.task_pos_weights.to(args['device']),
                                           reduction='none')
    optimizer = Adam(model.parameters(), lr=args['lr'])
    stopper = EarlyStopping(patience=args['patience'])
    model.to(args['device']) 

    epochx = 0 
    losses = [] 
    for epoch in range(args['num_epochs']): 
        # Train
        loss = run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        losses.append(loss) 

        # Validation and early stop
        epochx += 1 
        val_score = run_an_eval_epoch(args, model, val_loader, epochx, False)  
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric_name'],
            val_score, args['metric_name'], stopper.best_score))
        if early_stop:
            break

    stopper.load_checkpoint(model)

    # Print out the test set score 
     
    test_score = run_an_eval_epoch(args, model, test_loader, epochx, True)
    print('test {} {:.4f}'.format(args['metric_name'], test_score))
    
    # Making the loss per epoch figure 
    
    #print('losses', len(losses)) 
    print(losses) 
    epoch_list = [i+1 for i in range(len(losses))] ## 
    plt.clf() 
    plt.plot(epoch_list, losses) 
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
    plt.rcParams['axes.facecolor'] = 'white'
    plt.savefig("Loss.Per.Epoch.png") 


if __name__ == '__main__':
    args = {} 
    args['dataset'] = 'Tox21' 
    args['model'] = 'GAT' 
    args['exp'] = 'GAT_Tox21' 
    experimental_config = {
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100, ##
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'classifier_hidden_feats': 64,
    'num_heads': [4, 4],
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
    }
    args.update(experimental_config) 
    main(args)
