import sys
import os
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from dataloader import DataLoader
from fcx_vae_model import FCX_VAE
from blackboxmodel import BlackBox
from helpers import load_adult_income_dataset
from causal_modules import causal_regularization_enhanced, binarize_adj_matrix, ensure_dag
from LOFLoss import LOFLoss

cuda = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compute_loss( model, model_out, x, target_label, normalise_weights, validity_reg, margin,adj_matrix,pred_model=None ): 
    """
    Compute the combined ELBO, validity hinge-loss, sparsity penalty,
    and causal regularization for a batch of examples.

    Args:
        model (FCX_VAE): The VAE counterfactual generator.
        model_out (dict): Output of `model.forward`, containing:
            - em (Tensor): encoder means, shape (batch, latent_dim)
            - ev (Tensor): encoder variances, shape (batch, latent_dim)
            - z (Tensor): latent samples, list length mc_samples
            - x_pred (list[Tensor]): reconstructions per MC sample
            - mc_samples (int): number of Monte Carlo draws
        x (Tensor): Original input features, shape (batch, d).
        target_label (Tensor): True class labels (0 or 1), shape (batch,).
        normalise_weights (dict[int, tuple(float, float)]):
            Mapping feature index → (min, max) for proximity weighting.
        validity_reg (float): Weight of the validity hinge-loss term.
        margin (float): Hinge margin hyperparameter.
        adj_matrix (Tensor): Binary causal adjacency matrix, shape (d, d).

    Returns:
        Tensor: Scalar loss combining:
            - reconstruction error (L1 distance on mutable features)
            - KL divergence
            - validity hinge‑loss
            - sparsity penalty
            - causal regularization
    """
    lambda_nc = 1
    lambda_c = 1

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
    # immutables
    temp_copy = x.clone()
    temp_copy[:,:-4] = x_pred
    x_pred = temp_copy

    reg_loss = causal_regularization_enhanced(x_pred[:,:-4], adj_matrix,
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
        temp_copy[:,:-4] = x_pred
        x_pred = temp_copy
        reg_loss+=causal_regularization_enhanced(x_pred[:,:-4], adj_matrix,
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
        
        
        temp_logits = pred_model(x_pred)
        temp_1= temp_logits[target_label==1,:]
        temp_0= temp_logits[target_label==0,:]
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]).to(cuda) - F.sigmoid(temp_1[:,0]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]).to(cuda) - F.sigmoid(temp_0[:,1]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')

    
    recon_err = recon_err / mc_samples
    validity_loss = -1*validity_reg*validity_loss/mc_samples
    reg_loss=reg_loss/mc_samples

    sparsity = 1*1*sparsity
    
    print('recon: ',-torch.mean(recon_err), ' KL: ', torch.mean(kl_divergence), ' Validity: ', -validity_loss,'sparsity: ',sparsity,'reg_loss: ',reg_loss)
    #return -torch.mean(recon_err - kl_divergence) - validity_loss + sparsity +reg_loss*5 #20 #den kserw mhpws thelei meion
    loss = (
        -torch.mean(recon_err - kl_divergence)
        - validity_loss
        + sparsity
        + reg_loss * 5
    )
    # ensure it’s a 0‑d tensor, not a 1‑element vector
    return loss.squeeze()
    


def train_constraint_loss(model, train_dataset, optimizer, normalise_weights, validity_reg, constraint_reg, margin, epochs=1000, batch_size=1024,adj_matrix=None,ed_dict=None,pred_model=None):
    """
    Perform one epoch of FCX‑VAE training under causal and LOF constraints.

    This function iterates over the provided training data, computes the combined
    VAE ELBO loss, a causal constraint hinge loss education features,
    and a Local Outlier Factor (LOF) anomaly penalty on the latent codes. It then
    backpropagates and updates the model parameters.

    Args:
        model (FCX_VAE): The counterfactual VAE model to train.
        train_dataset (ndarray): Array of shape (N, D+1), where the last column is
            the binary label used for counterfactual conditioning.
        optimizer (torch.optim.Optimizer): Optimizer instance for model parameters.
        normalise_weights (dict[int, tuple[float, float]]): Per-feature (min, max)
            normalization weights for continuous features.
        validity_reg (float): Weight for the validity hinge loss component.
        constraint_reg (float): Weight for the causal/LOF constraint loss.
        margin (float): Margin hyperparameter for hinge losses.
        epochs (int, optional): Number of times to sample each datapoint (MC samples).
            Defaults to 1000.
        batch_size (int, optional): Mini-batch size for training. Defaults to 1024.
        adj_matrix (torch.Tensor, optional): Binary adjacency matrix enforcing causal
            dependencies. Shape (D, D).
        ed_dict (dict[int, int], optional): Mapping from categorical feature index to
            an education-level penalty coefficient.

    Returns:
        float: The sum of training losses over all mini-batches.
    """
    batch_num=0
    train_loss=0.0
    train_size=0
    criterion=LOFLoss(n_neighbors=30)    

    train_dataset= torch.tensor( train_dataset ).float().to(cuda)
    train_dataset= torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    lof_loss=0
    good_cf_count=0
    for train_x in enumerate(train_dataset):
        optimizer.zero_grad()

        train_x= train_x[1]
        train_x_back = train_x.clone()
        train_y = 1.0-torch.argmax( pred_model(train_x), dim=1 )
        train_x = train_x[:,:-4]
        train_size += train_x.shape[0]

        out= model(train_x, train_y)

        train_x_back[:,:-4] = train_x
        train_x= train_x_back

        loss = compute_loss(model, out, train_x, train_y, normalise_weights, validity_reg, margin,adj_matrix,pred_model)           

        dm = out['x_pred']
        mc_samples = out['mc_samples']
        x_pred = dm[0]

        #Age not decreasing
        constraint_loss = F.hinge_embedding_loss( x_pred[:,0] - train_x[:,0], torch.tensor(-1).to(cuda), 0).to(cuda)
        #Ed should not decrease
        for key in ed_dict.keys():
            constraint_loss += -0.05*ed_dict[key]*torch.mean( x_pred[:,key] - train_x[:,key] )

        for j in range(1, mc_samples):
            x_pred = dm[j]            
            constraint_loss+= F.hinge_embedding_loss( x_pred[:,0] - train_x[:,0], torch.tensor(-1).to(cuda), 0).to(cuda)           
            for key in ed_dict.keys():
                constraint_loss += -0.05*ed_dict[key]*torch.mean( x_pred[:,key] - train_x[:,key] )
        
        z_t = out['z']
        temp_lof_loss=0
        for z_temp in z_t:
            temp_lof_loss += criterion(z_temp)
        
        temp_lof_loss= temp_lof_loss/mc_samples

        constraint_loss= constraint_loss/mc_samples
        constraint_loss= constraint_reg*(constraint_loss)

        lof_loss=temp_lof_loss
        loss=loss+lof_loss*80

        loss=loss+ torch.mean(constraint_loss)
        train_loss += loss.item()
        batch_num+=1
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        else:
            pass

    ret= train_loss
    print('Train Avg Loss: ', ret, train_size)
    return ret


def train_binary_fcx_vae(
    dataset_name: str,
    base_data_dir: str = 'data/',
    base_model_dir: str = 'models/',
    batch_size: int = 64,
    epochs: int = 50,
    validity: float = 20.0,
    feasibility: float = 1.0,
    margin: float = 0.5,
):
    """
    Train the FCX‑VAE model to generate binary counterfactuals for the specified dataset.

    This function performs the full training pipeline:
    1. Sets random seeds and device.
    2. Builds an education-level penalty dictionary.
    3. Loads and preprocesses the Adult dataset, filtering only low-income samples.
    4. Loads normalization weights and the pretrained black-box classifier.
    5. Constructs the causal adjacency matrix and ensures it is a DAG.
    6. Initializes the FCX-VAE model and its optimizer.
    7. Runs a validity + feasibility constrained training loop for the given number of epochs.
    8. Saves the best model checkpoint and returns the trained VAE.

    Args:
        dataset_name (str): Dataset identifier, e.g. `'adult'`.
        base_data_dir (str): Path to the directory containing 
            `{dataset_name}-train-set.npy`, `{dataset_name}-val-set.npy`,
            normalization JSONs, and adjacency CSVs.
        base_model_dir (str): Directory to load the black‐box model from
            and save the final VAE checkpoint.
        batch_size (int): Mini-batch size for VAE training.
        epochs (int): Number of training epochs.
        validity (float): Weight for the validity hinge‐loss term.
        feasibility (float): Weight for the causal feasibility regularizer.
        margin (float): Margin parameter for all hinge losses.

    Returns:
        FCX_VAE: The trained VAE model instance, with its `state_dict` saved to
        `{base_model_dir}/{dataset_name}-margin-{margin}-feasibility-{feasibility}-validity-{validity}-epoch-{epochs}-fcx-binary.pth`.
    """
    constraint_reg=feasibility
    # Globals and reproducibility
    #global pred_model
    torch.manual_seed(10000000)
    global cuda, ed_dict
    cuda = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Encoded features & education mapping
    encoded_feature_names = ['age','hours_per_week','workclass_Government','workclass_Other/Unknown',
        'workclass_Private','workclass_Self-Employed','education_Assoc','education_Bachelors',
        'education_Doctorate','education_HS-grad','education_Masters','education_Prof-school',
        'education_School','education_Some-college','marital_status_Divorced',
        'marital_status_Married','marital_status_Separated','marital_status_Single',
        'marital_status_Widowed','occupation_Blue-Collar','occupation_Other/Unknown',
        'occupation_Professional','occupation_Sales','occupation_Service','occupation_White-Collar',
        'race_Other','race_White','gender_Female','gender_Male']
    education_score = {'HS-grad':0,'School':0,'Bachelors':1,'Assoc':1,'Some-college':1,'Masters':2,'Prof-school':2,'Doctorate':3}
    ed_dict = {encoded_feature_names.index(f'education_{k}'):v for k,v in education_score.items()}

    # 1. Data prep
    dataset = load_adult_income_dataset()
    params= {'dataframe':dataset.copy(), 'continuous_features':['age','hours_per_week'], 'outcome_name':'income'}
    d = DataLoader(params)  
    feat_to_change = d.get_indexes_of_features_to_vary(['age','hours_per_week','workclass','education','marital_status','occupation'])

    vae_train_dataset= np.load(base_data_dir+dataset_name+'-train-set.npy')
    vae_val_dataset= np.load(base_data_dir+dataset_name+'-val-set.npy')

    # CF Generation for only low to high income data points
    if dataset_name == 'adult':
        vae_train_dataset= vae_train_dataset[vae_train_dataset[:,-1]==0,:]
        vae_val_dataset= vae_val_dataset[vae_val_dataset[:,-1]==0,:]

    vae_train_dataset= vae_train_dataset[:,:-1]
    vae_val_dataset= vae_val_dataset[:,:-1]

    with open(base_data_dir+dataset_name+'-normalise_weights.json') as f:
        normalise_weights= json.load(f)
    normalise_weights = {int(k):v for k,v in normalise_weights.items()}

    with open(base_data_dir+dataset_name+'-mad.json') as f:
        mad_feature_weights= json.load(f)



    # GRAPH LOADING and ADJACENCY MATRIX
    dd_check = pd.read_csv(base_data_dir+'adult-train-set_check.csv')
    adj = pd.read_csv(base_data_dir+'adult_causal_graph_adjacency_matrix.csv',index_col=0)
    adj = adj.reindex(index=dd_check.columns, columns=dd_check.columns)

    if 'income' in adj.columns:
        adj = adj.drop('income',axis=0)
        adj = adj.drop('income',axis=1)

    adj = adj.drop('gender_Male',axis=0)
    adj = adj.drop('gender_Male',axis=1)

    adj = adj.drop('gender_Female',axis=0)
    adj = adj.drop('gender_Female',axis=1)

    adj = adj.drop('race_Other',axis=0)
    adj = adj.drop('race_Other',axis=1)

    adj = adj.drop('race_White',axis=0)
    adj = adj.drop('race_White',axis=1)

    # Create a directed graph
    G = nx.from_numpy_matrix(adj.values, create_using=nx.DiGraph())

    # Check for cycles
    try:
        topo_order = nx.topological_sort(G)
        print("Topological Order:", list(topo_order))
    except nx.NetworkXUnfeasible:
        print("The graph has at least one cycle.")

    # Visualize the graph
    #nx.draw(G, with_labels=True, arrows=True)
    #plt.show()

    # Create a directed graph
    G = nx.from_numpy_matrix(adj.values, create_using=nx.DiGraph())

    # Detect cycles
    try:
        cycles = list(nx.find_cycle(G, orientation='original'))
        print("Cycles found:", cycles)
        # Remove the last edge in the cycle to break it
        if cycles:
            edge_to_remove = cycles[-1][0], cycles[-1][1]
            G.remove_edge(*edge_to_remove)
            print(f"Removed edge: {edge_to_remove}")
    except nx.exception.NetworkXNoCycle:
        print("No cycles found.")


    adj2 = nx.to_numpy_matrix(G).astype(int)
    adj = pd.DataFrame(adj2, index=adj.columns, columns=adj.columns)
    adj_values = adj.values
    adj_values = binarize_adj_matrix(adj_values, threshold=0.5)
    adj_values = ensure_dag(adj_values)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj_values = torch.tensor(adj_values).float().to(device)


    #Load Black Box Model
    data_size= len(d.encoded_feature_names)
    pred_model= BlackBox(data_size).to(cuda)
    path= base_model_dir + dataset_name +'.pth'
    pred_model.load_state_dict(torch.load(path))
    pred_model.eval()

    # VAE MODEL
    wm1=1e-2
    wm2=1e-2
    wm3=1e-2
    wm4=1e-2
    data_size= len(feat_to_change)
    encoded_size=10

    fcx_vae = FCX_VAE(data_size, encoded_size, d).to(cuda)

    learning_rate = 1e-2

    fcx_vae_optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, fcx_vae.encoder_mean.parameters()),'weight_decay': wm1},
        {'params': filter(lambda p: p.requires_grad, fcx_vae.encoder_var.parameters()),'weight_decay': wm2},
        {'params': filter(lambda p: p.requires_grad, fcx_vae.decoder_mean.parameters()),'weight_decay': wm3}
    ], lr=learning_rate)

    #Train VAE
    loss_val = []
    likelihood_val = []
    valid_cf_count = []


    patience = 0
    epoch_time_list = []
    for epoch in range(epochs):
        
        np.random.shuffle(vae_train_dataset)
        start_time = time.time()

        loss_val.append( train_constraint_loss( fcx_vae, vae_train_dataset, fcx_vae_optimizer, normalise_weights, validity, constraint_reg, margin, 1, batch_size,adj_values,ed_dict,pred_model) )
        end_time = time.time()
        epoch_time_list.append(end_time-start_time)

        if epoch==0:
            best_loss= loss_val[-1]
        else:
            if loss_val[-1]-best_loss<5:
                best_loss= loss_val[-1]
                patience=0
                #Saving the final model
                torch.save(fcx_vae.state_dict(),  base_model_dir + dataset_name + '-margin-' + str(margin)  + '-feasibility-' + str(feasibility) + '-validity-'+ str(validity) + '-epoch-' + str(epochs) + '-' + 'fcx-binary' + '.pth')

            else:
                patience+=1

        print('----Epoch: ', epoch, ' Loss: ', loss_val[-1], ' Best: ', best_loss)
        if patience>5:# or epoch==12:
            break


    #mean time
    print("Mean time: ",np.mean(epoch_time_list))
    #sum time
    print("Sum time: ",np.sum(epoch_time_list))
    
    #Saving the final model
    torch.save(fcx_vae.state_dict(),  base_model_dir + dataset_name + '-margin-' + str(margin)  + '-feasibility-' + str(feasibility) + '-validity-'+ str(validity) + '-epoch-' + str(epochs) + '-' + 'fcx-binary' + '.pth')

    # plot loss val
    """
    plt.plot(loss_val)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()"""


    return fcx_vae

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='adult')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--validity', type=float, default=20)
    parser.add_argument('--feasibility', type=float, default=1)
    parser.add_argument('--margin', type=float, default=0.5)
    args = parser.parse_args()
    train_binary_fcx_vae(
        args.dataset_name,
        base_data_dir='data/',
        base_model_dir='models/',
        batch_size=args.batch_size,
        epochs=args.epoch,
        validity=args.validity,
        feasibility=args.feasibility,
        margin=args.margin,
        constraint_reg=args.constraint_reg
    )
