
import warnings
warnings.filterwarnings("ignore")

from scripts.dataloader import DataLoader
from scripts.fcx_vae_model import FCX_VAE
from scripts.blackboxmodel import BlackBox
from scripts.helpers import *
from scripts.causal_modules import causal_regularization_enhanced
from LOFLoss import LOFLoss
from scripts.causal_modules import binarize_adj_matrix, ensure_dag

import sys
import random
import pandas as pd
import numpy as np
import json
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

torch.manual_seed(10000000)
cuda= torch.device('cuda:0')

def compute_loss( model, model_out, x, target_label, normalise_weights, validity_reg, margin,adj_matrix ): 
    lambda_nc=1
    lambda_c=1
    em = model_out['em']
    ev = model_out['ev']
    z  = model_out['z']
    dm = model_out['x_pred']
    mc_samples = model_out['mc_samples']
            
    #KL Divergence
    kl_divergence = 0.5*torch.mean( em**2 +ev - torch.log(ev) - 1, axis=1 ) 
    #Reconstruction Term
    #Proximity: L1 Loss
    x_pred = dm[0]   
    temp_copy = x.clone()
    temp_copy[:,:-7] = x_pred
    x_pred = temp_copy
    reg_loss=causal_regularization_enhanced(x_pred[:,:-7], adj_matrix,
                                                  lambda_nc=lambda_nc, 
                                                  lambda_c=lambda_c)
    
    s= model.encoded_start_cat
    recon_err = -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
    for key in normalise_weights.keys():
        recon_err+= -(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 

    # Sum to 1 over the categorical indexes of a feature
    for v in model.encoded_categorical_feature_indexes:
        temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
        recon_err += temp

    count=0
    count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
    count+= torch.sum(x_pred[:,:s]>1,axis=1).float()    
    
    #Validity         
    temp_logits = pred_model(x_pred)
    validity_loss= torch.zeros(1).to(cuda)
    temp_1= temp_logits[target_label==1,:]
    temp_0= temp_logits[target_label==0,:]
    validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]).to(cuda) - F.sigmoid(temp_1[:,0]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
    validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]).to(cuda) - F.sigmoid(temp_0[:,1]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
    
    sparsity=torch.zeros(1).to(cuda)
    for sample in range(0,x.shape[0]):
        temp=0
        for v in model.encoded_categorical_feature_indexes[:-2]:
            temp +=0.5*torch.sum( torch.sum( torch.norm(x_pred[sample, v[0]:v[-1]+1]-x[sample, v[0]:v[-1]+1],p=1)>0.01) )#/x.shape[0]

        
        for t in [0,1]:
            temp +=0.5*torch.sum( torch.sum( torch.norm(x_pred[sample, t]-x[sample, t])) )

        sparsity += temp

    sparsity =1*(sparsity/x.shape[0])
    

    for i in range(1,mc_samples):
        x_pred = dm[i]        
        # immutable variables at the end
        temp_copy = x.clone()
        temp_copy[:,:-7] = x_pred
        x_pred = temp_copy
        reg_loss+=causal_regularization_enhanced(x_pred[:,:-7], adj_matrix,
                                                  lambda_nc=lambda_nc, 
                                                  lambda_c=lambda_c)

        recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
        for key in normalise_weights.keys():
            recon_err+= -(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 
            
        # Sum to 1 over the categorical indexes of a feature
        for v in model.encoded_categorical_feature_indexes:
            temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
            recon_err += temp

        count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
        count+= torch.sum(x_pred[:,:s]>1,axis=1).float()        
            
        #Validity
        temp_logits = pred_model(x_pred)
        temp_1= temp_logits[target_label==1,:]
        temp_0= temp_logits[target_label==0,:]
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]).to(cuda) - F.sigmoid(temp_1[:,0]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]).to(cuda) - F.sigmoid(temp_0[:,1]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')

    reg_loss=reg_loss/mc_samples
    recon_err = recon_err / mc_samples
    validity_loss = -1*validity_reg*validity_loss/mc_samples
    sparsity = 1*1*sparsity

    print('recon: ',-torch.mean(recon_err), ' KL: ', torch.mean(kl_divergence), ' Validity: ', -validity_loss, ' Sparsity: ', sparsity, ' Reg loss: ', reg_loss)
    return -torch.mean(recon_err - kl_divergence) - validity_loss + sparsity + 5*reg_loss    
    

def train_constraint_loss(model, train_dataset, optimizer, normalise_weights, validity_reg, constraint_reg, margin, epochs=1000, batch_size=1024,adj_matrix=None):
    batch_num=0
    train_loss=0.0
    train_size=0
    criterion=LOFLoss(n_neighbors=50) #htan 30

    train_dataset= torch.tensor( train_dataset ).float().to(cuda)
    train_dataset= torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    
    good_cf_count=0
    for train_x in enumerate(train_dataset):
        optimizer.zero_grad()

        train_x= train_x[1]
        train_x_back = train_x.clone()
        train_y = 1.0-torch.argmax( pred_model(train_x), dim=1 )
        train_x = train_x[:,:-7]
        train_size += train_x.shape[0]

        out= model(train_x, train_y)

        train_x_back[:,:-7] = train_x
        train_x= train_x_back

        loss = compute_loss(model, out, train_x, train_y, normalise_weights, validity_reg, margin,adj_matrix)           
        
        dm = out['x_pred']
        mc_samples = out['mc_samples']
        x_pred = dm[0]

        constraint_loss = F.hinge_embedding_loss( x_pred[:,0] - train_x[:,0], torch.tensor(-1).to(cuda), 0).to(cuda)

        for j in range(1, mc_samples):
            x_pred = dm[j]            
            constraint_loss+= F.hinge_embedding_loss( x_pred[:,0] - train_x[:,0], torch.tensor(-1).to(cuda), 0).to(cuda)       
   
        constraint_loss= constraint_loss/mc_samples
        
        constraint_loss= constraint_reg*constraint_loss
        z_t = out['z']
        z_t=z_t[0]
        lof_loss = criterion(z_t)

        loss+=lof_loss*(1e-6)#40

        loss+= torch.mean(constraint_loss)
        train_loss += loss.item()
        batch_num+=1
        
        loss.backward(retain_graph=True) #retain_graph=True
        optimizer.step()        

    ret= train_loss
    print('Train Avg Loss: ', ret, train_size)
    return ret


def train_unary_fcx_vae(
    dataset_name: str,
    base_data_dir='data/',
    base_model_dir='models/',
    batch_size: int = 64,
    epochs: int = 50,
    validity: float = 20,
    feasibility: float = 1,
    margin: float = 0.5
):
    # Seed for reproducibility
    global pred_model,cuda
    torch.manual_seed(10000000)
    # GPU
    cuda = torch.device('cuda:0')
    # Dataset
    dataset = pd.read_csv(
        os.path.join(base_data_dir, 'census', 'census_data.csv')
    )
    params = {
        'dataframe': dataset.copy(),
        'continuous_features': [
            'age','wage_per_hour','capital_gains','capital_losses',
            'dividends_from_stocks','num_persons_worked_for_employer','weeks_worked_in_year'
        ],
        'outcome_name': 'income'
    }
    d = DataLoader(params)
    feat_to_change = d.get_indexes_of_features_to_vary([
        'age','class_of_worker','major_industry_code','major_occupation_code',
        'education','wage_per_hour','marital_status','full_parttime_employment_stat',
        'capital_gains','capital_losses','dividends_from_stocks','tax_filer_status',
        'd_household_family_stat','d_household_summary','num_persons_worked_for_employer',
        'family_members_under_18','citizenship','own_business_or_self_employed',
        'veterans_benefits','weeks_worked_in_year','year'
    ])

    vae_train_dataset = np.load(
        os.path.join(base_data_dir, 'census', f"{dataset_name}-train-set.npy")
    )
    vae_val_dataset   = np.load(
        os.path.join(base_data_dir, 'census', f"{dataset_name}-val-set.npy")
    )

    # CF Generation for only low to high income
    vae_train_dataset = vae_train_dataset[vae_train_dataset[:,-1] == 0, :]
    vae_val_dataset   = vae_val_dataset[vae_val_dataset[:,-1] == 0, :]
    vae_train_dataset = vae_train_dataset[:, :-1]
    vae_val_dataset   = vae_val_dataset[:, :-1]

    with open(
        os.path.join(base_data_dir, 'census', f"{dataset_name}-normalise_weights.json")
    ) as f:
        normalise_weights = {int(k):v for k,v in json.load(f).items()}

    with open(
        os.path.join(base_data_dir, 'census', f"{dataset_name}-mad.json")
    ) as f:
        mad_feature_weights = json.load(f)

    # Load Black Box Model
    data_size   = len(d.encoded_feature_names)
    pred_model  = BlackBox(data_size).to(cuda)
    path        = os.path.join(base_model_dir, dataset_name + '.pth')
    pred_model.load_state_dict(torch.load(path))
    pred_model.eval()

    # GRAPH LOADING and ADJACENCY MATRIX
    dd_check = pd.read_csv(
        os.path.join(base_data_dir, 'census', f"{dataset_name}-train-set_check.csv")
    )
    adj = pd.read_csv(
        os.path.join(base_data_dir, 'census', f"{dataset_name}_causal_graph_adjacency_matrix.csv"),
        index_col=0
    )
    adj = adj.reindex(index=dd_check.columns, columns=dd_check.columns)

    # Drop income and immutable cols
    for col in ['income', 'sex_Male', 'sex_Female', 'race_Other',
                'race_White', 'race_Black', 'race_Asian or Pacific Islander',
                'race_Amer Indian Aleut or Eskimo']:
        if col in adj.columns:
            adj = adj.drop(col, axis=0).drop(col, axis=1)

    G = nx.from_numpy_matrix(adj.values, create_using=nx.DiGraph)
    try:
        topo_order = nx.topological_sort(G)
        print("Topological Order:", list(topo_order))
    except nx.NetworkXUnfeasible:
        print("The graph has at least one cycle.")

    # Detect cycles and remove one edge
    try:
        cycles = list(nx.find_cycle(G, orientation='original'))
        if cycles:
            G.remove_edge(cycles[-1][0], cycles[-1][1])
    except nx.exception.NetworkXNoCycle:
        pass

    adj2       = nx.to_numpy_matrix(G).astype(int)
    adj_df     = pd.DataFrame(adj2, index=adj.columns, columns=adj.columns)
    adj_values = binarize_adj_matrix(adj_df.values, threshold=0.5)
    adj_values = ensure_dag(adj_values)
    adj_values = torch.tensor(adj_values).float().to(cuda)

    # Initialise VAE model
    wm1 = wm2 = wm3 = wm4 = 1e-2
    data_size    = len(feat_to_change)
    encoded_size = 30
    fcx_vae      = FCX_VAE(data_size, encoded_size, d).to(cuda)
    learning_rate= 1e-1
    fcx_vae_optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, fcx_vae.encoder_mean.parameters()), 'weight_decay': wm1},
        {'params': filter(lambda p: p.requires_grad, fcx_vae.encoder_var.parameters()),  'weight_decay': wm2},
        {'params': filter(lambda p: p.requires_grad, fcx_vae.decoder_mean.parameters()),'weight_decay': wm3}
    ], lr=learning_rate)

    # Optional random sample dataset
    vae_train_dataset = vae_train_dataset[
        np.random.choice(vae_train_dataset.shape[0], 100000, replace=False), :
    ]

    loss_val = []
    for ep in range(epochs):
        np.random.shuffle(vae_train_dataset)
        loss_ep = train_constraint_loss(
            fcx_vae, vae_train_dataset, fcx_vae_optimizer,
            normalise_weights, validity, feasibility, margin,
            1, batch_size, adj_values
        )
        loss_val.append(loss_ep)
        print(f"epoch {ep} loss {loss_ep}")

    # Saving the final model
    torch.save(
        fcx_vae.state_dict(),
        os.path.join(
            base_model_dir,
            f"{dataset_name}-margin-{margin}-feasibility-{feasibility}-validity-{validity}-epoch-{epochs}-fcx-unary.pth"
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,   default='census')
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--epochs',         type=int,   default=50)
    parser.add_argument('--validity',      type=float, default=20)
    parser.add_argument('--feasibility',   type=float, default=1)
    parser.add_argument('--margin',        type=float, default=0.5)
    args = parser.parse_args()

    train_unary_fcx_vae(
        args.dataset_name,
        base_data_dir='data/',
        base_model_dir='models/',
        batch_size=args.batch_size,
        epochs=args.epochs,
        validity=args.validity,
        feasibility=args.feasibility,
        margin=args.margin
    )