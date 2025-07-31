# ------------------- LIBRARIES ------------------- 
from scripts.fcx_vae_model import FCX_VAE

import sys
import random
import pandas as pd
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
#Seed for repoduability
torch.manual_seed(10000000)
from sklearn.neighbors import LocalOutlierFactor

offset=0.0
# ------------------- FEATURE NAMES ------------------- 

education_change= {'HS-grad': ['Some-college', 'Bachelors', 'Assoc', 'Prof-school', 'Masters', 'Doctorate'],
        'School': ['Some-college', 'Bachelors', 'Assoc', 'Prof-school', 'Masters', 'Doctorate'],
        'Bachelors': ['Prof-school', 'Masters', 'Doctorate'],
        'Assoc': ['Prof-school', 'Masters', 'Doctorate'] ,
        'Some-college': ['Prof-school', 'Masters', 'Doctorate'],
        'Masters': ['Doctorate'],
        'Prof-school': ['Doctorate'],
        'Doctorate': [] }

education_score= {'HS-grad': 0,
        'School': 0,
        'Bachelors': 1,
        'Assoc': 1,
        'Some-college': 1,
        'Masters': 2,
        'Prof-school': 2,
        'Doctorate': 3 }

encoded_feature_names=['age',
 'hours_per_week',
 'workclass_Government',
 'workclass_Other/Unknown',
 'workclass_Private',
 'workclass_Self-Employed',
 'education_Assoc',
 'education_Bachelors',
 'education_Doctorate',
 'education_HS-grad',
 'education_Masters',
 'education_Prof-school',
 'education_School',
 'education_Some-college',
 'marital_status_Divorced',
 'marital_status_Married',
 'marital_status_Separated',
 'marital_status_Single',
 'marital_status_Widowed',
 'occupation_Blue-Collar',
 'occupation_Other/Unknown',
 'occupation_Professional',
 'occupation_Sales',
 'occupation_Service',
 'occupation_White-Collar',
 'race_Other',
 'race_White',
 'gender_Female',
 'gender_Male']
ed_dict={}
for key in education_score.keys():
    ed_dict[encoded_feature_names.index('education_'+key)]= education_score[key]
print(ed_dict)


# ------------------- HELPER FUNCTIONS ------------------- 
    
def de_normalise( x, normalise_weights ):
    """
    Map a normalized feature value back to its original scale.

    Args:
        x (float or array-like):
            The normalized value(s) in the range [0, 1].
        normalise_weights (tuple[float, float]):
            A `(min, max)` tuple giving the original feature’s range.

    Returns:
        float or array-like:
            The de-normalized value(s), scaled back to [min, max].
    """
    return (normalise_weights[1] - normalise_weights[0])*x + normalise_weights[0]

def de_scale( x, normalise_weights ):
    """
    Compute the absolute scale (range) of a normalized value.

    Given a normalized offset `x` in [0, 1], and a feature’s original
    `(min, max)` range, returns the corresponding absolute difference.

    Args:
        x (float):
            The normalized offset (e.g., 0.05 represents 5% of the range).
        normalise_weights (tuple[float, float]):
            A `(min, max)` tuple for the feature’s original range.

    Returns:
        float:
            The absolute offset in the original scale (i.e., `(max - min) * x`).
    """
    return (normalise_weights[1] - normalise_weights[0])*x



# ------------------- VALIDITY METRIC ------------------- 
def validity_score(model, pred_model, train_dataset, case, sample_range,d=None):

    """
    Compute the validity of generated counterfactuals as the percentage that
    successfully flip the black‑box classifier’s prediction.

    For each `sample_size` in `sample_range`, this function:
      1. Generates `sample_size` counterfactuals per example via `model.compute_elbo`,
         conditioned on the opposite class from `pred_model`.
      2. Applies `pred_model` to each generated counterfactual.
      3. Computes the fraction of counterfactuals whose predicted label
         differs from the original label.

    Args:
        model (FCX_VAE):
            Trained counterfactual VAE providing `compute_elbo(...)`.
        pred_model (BlackBox):
            Pre‑trained classifier used to judge validity of counterfactuals.
        train_dataset (np.ndarray):
            Array of original examples with labels, shape (N, d+1).
        case (int or bool):
            If truthy, may trigger optional plotting (unused by default).
        sample_range (list[int]):
            Monte Carlo sample counts to evaluate (e.g., [1, 2, 3]).
        d (DataLoader, optional):
            DataLoader instance for decoding or de-normalizing data
            (if needed).

    Returns:
        list[float]:
            Validity percentages (0–100)
    """
    cf_df = pd.DataFrame(columns=['x','label', 'x_cf',  'label_cf'])
    x_pred_list = []
    validity_score_arr=[]
    vector = np.array([])
    print("sample range: ", sample_range)
    #sample_range=[1]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y = torch.argmax( pred_model(train_x), dim=1 )                
        valid_cf_count=0
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label,z = model.compute_elbo( train_x, 1.0-train_y, pred_model,True )
            print("------------------------------------")
            print("sample iter ",sample_iter)

            #add x train and x pred to dataframe
            if d!=None:
                x_pred2= d.de_normalize_data( d.get_decoded_data(x_pred.cpu().detach().numpy()) )
                x_true2= d.de_normalize_data( d.get_decoded_data(x_true.cpu().detach().numpy()) )                
            
            x_pred2['Class'] = cf_label.cpu().detach().numpy()
            x_true2['Class'] = train_y.cpu().detach().numpy()
            vector = z.cpu().detach().numpy()

            cf_label = cf_label.numpy()
            valid_cf_count += np.sum( train_y.numpy() != cf_label )
            
        test_size= train_x.shape[0]
        valid_cf_count=valid_cf_count/sample_size
        validity_score_arr.append(100*valid_cf_count/test_size)
    
    # concatenate arrays in the list
    x_true2.to_csv("x_true_binary_lof.csv")
    x_pred2.to_csv("x_pred_binary_lof.csv")

    z = pd.DataFrame(vector)
    z.to_csv("z_pred__binary_lof.csv")

    """if case:
        plt.plot(sample_range, validity_score_arr)
        plt.title('Valid CF')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()"""
    print('Mean Validity Score: ', np.mean(np.array(validity_score_arr)) )
    return validity_score_arr

# ------------------- PROXIMITY METRIC ------------------- 
def proximity_score(model, pred_model, train_dataset, d, mad_feature_weights, cat, case, sample_range):
    """
    Compute continuous or categorical L1 proximity between originals and counterfactuals.

    For each `sample_size` in `sample_range`, this function:
      1. Generates `sample_size` counterfactuals per example via `model.compute_elbo`,
         conditioned to flip `pred_model`’s output.
      2. De-normalizes both originals (`x_true`) and counterfactuals (`x_pred`) using
         `d.get_decoded_data` and `d.de_normalize_data`.
      3. Computes L1 distances:
         - If `cat` is False: sums absolute differences over continuous features.
         - If `cat` is True: sums absolute differences over one-hot encoded categorical features.
      4. Averages distances across MC samples and examples.

    Args:
        model (FCX_VAE):
            Trained counterfactual VAE providing `compute_elbo(...)`.
        pred_model (BlackBox):
            Pre‑trained classifier for validity conditioning.
        train_dataset (np.ndarray):
            Array of original examples with labels, shape (N, d+1).
        d (DataLoader):
            DataLoader instance for decoding and de-normalizing features.
        mad_feature_weights (dict[int, float]):
            Mean absolute deviation weights for continuous features.
        cat (bool):
            If False, compute continuous proximity; if True, compute categorical proximity.
        case (int or bool):
            If truthy, may trigger plotting (unused by default).
        sample_range (list[int]):
            Monte Carlo sample counts to evaluate (e.g., [1, 2, 3]).

    Returns:
        list[float]:
            Proximity scores (average L1 distances)
    """
    prox_score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y = torch.argmax( pred_model(train_x), dim=1 )                
        prox_count=0
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label,z = model.compute_elbo( train_x, 1.0-train_y, pred_model ,True)            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )   
             
            if cat:
                for column in d.categorical_feature_names:
                    prox_count += np.sum( np.array(x_true[column], dtype=pd.Series) != np.array(x_pred[column], dtype=pd.Series ))
            else:
                for column in d.continuous_feature_names:
                    if column in ['wage_per_hour', 'capital_gains','capital_losses','dividends_from_stocks']:
                        continue
                    prox_count += np.sum(np.abs(x_true[column] - x_pred[column]))/mad_feature_weights[column]                
                    
        test_size= train_x.shape[0]
        prox_count= prox_count/sample_size
        prox_score_arr.append( -1*prox_count/test_size )

    if case:
        pass
        """plt.plot(sample_range, prox_score_arr)
        if cat:
            plt.title('Categorical Proximity')
        else:
            plt.title('Continuous Proximity')

        plt.xlabel('Sample Size')
        plt.ylabel('Magnitude')
        plt.show()"""
    print('Mean Proximity Score: ', np.mean(np.array(prox_score_arr)) )
    return prox_score_arr


# ------------------- BINARY CONSTRAINT ------------------- 
def causal_score_age_ed_constraint(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range):
    """
    Compute feasibility scores based on age and education monotonicity constraints for binary counterfactuals.

    For each `sample_size` in `sample_range`, this function:
      1. Generates `sample_size` counterfactuals per example via `model.compute_elbo`, conditioned to flip `pred_model`’s output.
      2. De-normalizes counterfactuals (`x_pred`) and originals (`x_true`) with `d.get_decoded_data` and `d.de_normalize_data`.
      3. Checks two monotonic constraints for each counterfactual:
         - **Age constraint**: CF age ≥ original age + de-scaled `offset`.
         - **Education constraint**: CF education level ≥ original education level + de-scaled `offset`.
      4. Computes the percentage of valid vs. invalid counterfactuals for each `sample_size`.

    Args:
        model (FCX_VAE):
            Trained counterfactual VAE model providing `compute_elbo(...)`.
        pred_model (BlackBox):
            Pre‑trained classifier used to determine target labels.
        train_dataset (np.ndarray):
            Array of original examples with labels, shape (N, d+1).
        d (DataLoader):
            DataLoader instance for decoding and de-normalizing features.
        normalise_weights (dict[int, tuple(float, float)]):
            Per-feature (min, max) values for scaling/offset computations.
        offset (float):
            Raw offset applied when checking monotonic constraints.
        case (int or bool):
            If truthy, may trigger optional plotting (unused by default).
        sample_range (list[int]):
            Monte Carlo sample counts to evaluate (e.g., [1, 2, 3]).

    Returns:
        tuple:
            - valid_score_arr (list[float]): Percentage of counterfactuals satisfying
              both age and education constraints, for each `sample_size`.
            - invalid_score_arr (list[float]): Percentage violating at least one
              constraint, for each `sample_size`.
    """
    valid_score_arr=[]
    invalid_score_arr=[]
    count1=0
    count2=0
    count3=0
    pos1=0
    pos2=0
    pos3=0
    
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change= 0
        invalid_change=0
        test_size=0
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label,z= model.compute_elbo( train_x, 1.0-train_y, pred_model ,True)            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            ed_idx = x_true.columns.get_loc('education')
            age_idx = x_true.columns.get_loc('age')            

            for i in range(x_true.shape[0]): 
                
                if cf_label[i]==0:
                    continue                
                test_size+=1
                
                if education_score[ x_pred.iloc[i,ed_idx] ] < education_score[ x_true.iloc[i,ed_idx] ]:
                    count3+=1
                    invalid_change +=1       
                elif education_score[ x_pred.iloc[i,ed_idx] ] == education_score[ x_true.iloc[i,ed_idx] ]:
                    count1+=1
                    if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0]) >= x_true.iloc[i, age_idx]:
                        pos1+=1
                        valid_change += 1
                    else:
                        invalid_change +=1                    
                elif education_score[ x_pred.iloc[i,ed_idx] ] > education_score[ x_true.iloc[i,ed_idx] ]:
                    count2+=1
                    if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0]) > x_true.iloc[i, age_idx]:
                        pos2+=1
                        valid_change += 1
                    else:
                        invalid_change +=1

        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size

        test_size= test_size/sample_size
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))

    if case:
        pass
        """plt.plot(sample_range, valid_score_arr, '*', label='Val Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('All Education Levels')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()"""
    print('Mean Age-Ed Constraint Score: ', valid_score, invalid_score, valid_score/(valid_score+invalid_score))
    print('Count: ', count1, count2, count3, count1+count2+count3)
    print('Pos Count: ', pos1, pos2, pos3 )
    if count1 and count2 and count3:
        print('Pos Percentage: ', pos1/count1, pos2/count2, pos3/count3 )    
    return valid_score_arr, invalid_score_arr


def lof_score_func(data):
    """
    Compute the average Local Outlier Factor (LOF) anomaly score for a dataset.

    This function fits a LocalOutlierFactor model (with 20 neighbors and
    Euclidean distance) on the input data and returns the mean LOF score
    across all samples. Higher LOF scores indicate a greater degree of
    anomaly.

    Args:
        data (array-like of shape (n_samples, n_features)):
            The input data on which to compute anomaly scores.

    Returns:
        float:
            The average LOF anomaly score (mean of `-negative_outlier_factor_`)
            over all samples.
    """
    scores = []
    n_outliers = []
    
    # Split the data into train and test sets
    #X_train, X_test = train_test_split(data, test_size=0.01, random_state=42)
    X_train=data
    k_list=[]
    # Initialize LOF with the current k value
    lof = LocalOutlierFactor(n_neighbors=20,metric='euclidean') #, contamination=contamination
    
    # Fit the LOF model and get the outlier scores
    y_pred = lof.fit_predict(X_train)
    lof_scores = -lof.negative_outlier_factor_  # Higher scores = more likely to be outliers
    
    # Calculate the number of outliers detected in the training set
    num_outliers = np.sum(y_pred == -1)
    
    # Average LOF score (mean anomaly score)
    avg_lof_score = np.mean(lof_scores)

    return avg_lof_score#, num_outliers

# ------------------- BINARY CONSTRAINT + LOF ------------------- 
def causal_score_age_ed_constraint_lof(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range,prefix_name='test'):
    """
    Compute a combined age and education constraint violation penalty with LOF anomaly score for binary counterfactuals.

    For each `sample_size` in `sample_range`, this function:
      1. Generates `sample_size` counterfactuals per example via `model.compute_elbo`, conditioned to flip `pred_model`’s output.
      2. De-normalizes the counterfactuals and originals using `d.get_decoded_data` and `d.de_normalize_data`.
      3. Evaluates two monotonic constraints:
         - **Age constraint**: CF age ≥ original age + de-scaled `offset`.
         - **Education constraint**: CF education level ≥ original education level (using `offset` and `normalise_weights`).
      4. Computes a Local Outlier Factor (LOF) anomaly score on the VAE’s latent codes.
      5. Combines the two constraint violation rates with the LOF score into a single numeric score per example.

    Args:
        model (FCX_VAE):
            Trained counterfactual VAE model.
        pred_model (BlackBox):
            Pre‑trained classifier used for validity conditioning.
        train_dataset (np.ndarray):
            Array of examples + labels, shape (N, d+1).
        d (DataLoader):
            DataLoader instance for de-coding and de-normalizing features.
        normalise_weights (dict[int, tuple(float, float)]):
            Per-feature (min, max) values for scaling.
        offset (float):
            Raw offset applied when checking monotonic constraints.
        case (int or bool):
            If truthy, may trigger plotting (unused by default).
        sample_range (list[int]):
            Monte Carlo sample counts to evaluate (e.g., [1, 2, 3]).
        prefix_name (str, optional):
            Prefix for any saved outputs or logs (default: 'test').

    Returns:
        list[float]:
            LOF score
    """
    valid_score_arr=[]
    invalid_score_arr=[]
    count1=0
    count2=0
    count3=0
    pos1=0
    pos2=0
    pos3=0
    full_lof=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change= 0
        invalid_change=0
        test_size=0
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label,z= model.compute_elbo( train_x, 1.0-train_y, pred_model ,True)
            #copy tensor
            x_pred_norm = x_pred.clone()         
            feas = []
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            ed_idx = x_true.columns.get_loc('education')
            age_idx = x_true.columns.get_loc('age')            

            for i in range(x_true.shape[0]): 
                
                if cf_label[i]==0:
                    #print(cf_label[i])
                    feas.append(-1)
                    continue                
                test_size+=1
                
                if education_score[ x_pred.iloc[i,ed_idx] ] < education_score[ x_true.iloc[i,ed_idx] ]:
                    count3+=1
                    invalid_change +=1
                    feas.append(0)       
                elif education_score[ x_pred.iloc[i,ed_idx] ] == education_score[ x_true.iloc[i,ed_idx] ]:
                    count1+=1
                    if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0]) >= x_true.iloc[i, age_idx]:
                        pos1+=1
                        valid_change += 1
                        feas.append(1)
                    else:
                        invalid_change +=1    
                        feas.append(0)                
                elif education_score[ x_pred.iloc[i,ed_idx] ] > education_score[ x_true.iloc[i,ed_idx] ]:
                    count2+=1
                    if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0]) > x_true.iloc[i, age_idx]:
                        pos2+=1
                        valid_change += 1
                        feas.append(1)
                    else:
                        invalid_change +=1
                        feas.append(0)
            feas = np.array(feas)
            num_all = feas.shape[0]
            
            x_pred_norm = x_pred_norm.detach().numpy()

            x_pred_norm = x_pred_norm[feas==1]
            vector = z.detach().numpy()
            z = vector[feas==1]
            lof_score_val = lof_score_func(x_pred_norm)
            #lof_score_val = lof_score_func(x_pred_norm)

            x_pred['Class'] = cf_label.cpu().detach().numpy()
            x_true['Class'] = train_y.cpu().detach().numpy()
            x_true.to_csv(prefix_name + "_x_true.csv",index=False)
            x_pred.to_csv(prefix_name + "_x_pred.csv",index=False)

            z = pd.DataFrame(vector)
            z.to_csv(prefix_name+"_z_pred.csv",index=False)
            xprednorm = pd.DataFrame(x_pred_norm)
            xprednorm.to_csv(prefix_name+"_x_pred_norm.csv",index=False)
            
            print("----->> Average lof score {}".format(lof_score_val))
            
            #plot with tsne import tsne
            #lof = LocalOutlierFactor(n_neighbors=6, metric='euclidean')
            #lof_scores= lof.fit_predict(x_pred_norm) # tha mporouse kai gia x_true na ypologistei


            # PLOT lof score for different k
            """k_lof_results=[]
            k_lof_values=[]
            for k in range(1,200,1):
                lof = LocalOutlierFactor(n_neighbors=k, metric='euclidean')
                lof_scores= lof.fit_predict(x_pred_norm) # tha mporouse kai gia x_true na ypologistei
                temp_lof_score = np.sum(lof_scores==-1)
                k_lof_values.append(np.mean(lof.negative_outlier_factor_))
                k_lof_results.append(temp_lof_score)
            
            k_range = [*range(1,200,1)]
            print("k_lof_results", k_lof_results)
            plt.figure()
            plt.plot(k_range, k_lof_values)
            plt.title('LOF CF')
            plt.xlabel('k')
            plt.ylabel('LOF')
            #plt.show()
            plt.savefig(prefix_name+"_lof_b.png")
            #save k_range and k_lof_results to pandas datafframe
            df = pd.DataFrame(list(zip(k_range, k_lof_results,k_lof_values)), columns =['k','num_outliers','lof'])
            df.to_csv(prefix_name+"_lof_k_b.csv",index=False)"""

            full_lof.append(lof_score_val)

        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size

        test_size= test_size/sample_size
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr)) 
    invalid_score= np.mean(np.array(invalid_score_arr))
    full_lof_score = np.mean(np.array(full_lof))
    print("Final average lof score", full_lof_score)
    return full_lof

# ------------------- UNARY CONSTRAINT ------------------- 
def causal_score_age_constraint(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range):
    """
    Compute feasibility scores based on a monotonic age constraint for counterfactuals.

    For each `sample_size` in `sample_range`, this function:
      1. Calls `model.compute_elbo` to generate `sample_size` counterfactuals per example.
      2. De-normalizes predictions (`x_pred`) and originals (`x_true`) via the `DataLoader`.
      3. Checks that the counterfactual age ≥ original age plus a scaled `offset`.
      4. Computes the percentage of valid and invalid age‐constraint satisfactions.

    Args:
        model (FCX_VAE):
            The trained counterfactual VAE, providing `compute_elbo(...)`.
        pred_model (BlackBox):
            Pre‑trained classifier used to determine target labels.
        train_dataset (np.ndarray):
            Array of shape (N, d+1), where the last column is the true label.
        d (DataLoader):
            DataLoader instance with methods:
              - `get_decoded_data(...)` to map back to original feature names,
              - `de_normalize_data(...)` to apply inverse scaling.
        normalise_weights (dict[int, tuple(float, float)]):
            Per‑feature (min, max) values for de-scaling.
        offset (float):
            Raw offset to subtract (via `de_scale(offset, normalise_weights[0])`)
            when comparing ages.
        case (bool or int):
            If truthy, plot valid vs invalid percentages (unused by default).
        sample_range (list[int]):
            List of Monte Carlo sample counts to evaluate (e.g., [1, 2, 3]).

    Returns:
        tuple:
            - valid_score_arr (list[float]): Percentage of counterfactuals
              satisfying the age constraint, for each `sample_size`.
            - invalid_score_arr (list[float]): Percentage violating the
              age constraint, for each `sample_size`.
    """
    valid_score_arr=[]
    invalid_score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change= 0
        invalid_change=0 
        test_size=0
        
        for sample_iter in range(sample_size):
            recon_err, kl_err, x_true, x_pred, cf_label,z = model.compute_elbo( train_x, 1.0-train_y, pred_model,True )            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            age_idx = x_true.columns.get_loc('age')            
            for i in range(x_true.shape[0]): 
                
                if cf_label[i]==0:
                    continue                
                test_size+=1
                
                if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0] ) >= x_true.iloc[i, age_idx]:
                    valid_change+=1
                else:
                    invalid_change+=1
        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size

        test_size= test_size/sample_size
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))

    if case:
        pass
        """plt.plot(sample_range, valid_score_arr, '*', label='Val Age Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Age Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in Age')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()    """
    print('Mean Age Constraint Score: ', valid_score, invalid_score, valid_score/(valid_score+invalid_score))
    return valid_score_arr, invalid_score_arr

# ------------------- UNARY CONSTRAINT + LOF ------------------- 
def causal_score_age_constraint_lof(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range,prefix_name='test'):
    """
    Compute a combined age-education-constraint violation penalty and LOF anomaly score for generated counterfactuals (LOF for feasible cf examples).

    For each sample in `train_dataset`, this function:
      1. Generates counterfactuals via the FCX‑VAE `model` conditioned to flip the
         `pred_model` label.
      2. Measures any violations of the monotonic age constraint relative to the
         original age feature (i.e. that age should not decrease).
      3. Computes a Local Outlier Factor (LOF) anomaly score on the latent encodings.
      4. Returns a single combined score per sample.

    Args:
        model (FCX_VAE): Trained counterfactual VAE.
        pred_model (BlackBox): Pre-trained classifier used for validity checks.
        train_dataset (np.ndarray):
            Original examples with labels, shape `(N, num_features+1)`.
        d (DataLoader):
            DataLoader instance providing feature splits and metadata.
        normalise_weights (dict[int, tuple(float, float)]):
            Per-feature (min, max) weights for proximity and scaling.
        offset (int):
            Index offset indicating how many initial features to treat as immutable
            (used to locate the age feature).
        case (int):
            Metric case identifier (currently unused).
        sample_range (list[int]):
            Indices of Monte Carlo samples to generate counterfactuals.
        prefix_name (str, optional):
            Prefix for any output logging or filenames (default: `'test'`).

    Returns:
        List[float]: LOF scores.
    """
    valid_score_arr=[]
    invalid_score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change= 0
        invalid_change=0 
        test_size=0
        full_lof=[]
        for sample_iter in range(sample_size):
            recon_err, kl_err, x_true, x_pred, cf_label,z = model.compute_elbo( train_x, 1.0-train_y, pred_model,True )            
            x_pred_norm = x_pred.clone()
            feas = []
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            age_idx = x_true.columns.get_loc('age')            
            for i in range(x_true.shape[0]): 
                
                if cf_label[i]==0:
                    feas.append(-1)
                    continue                
                test_size+=1
                
                if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0] ) >= x_true.iloc[i, age_idx]:
                    valid_change+=1
                    feas.append(1)
                else:
                    invalid_change+=1
                    feas.append(0)
            feas = np.array(feas)
            num_all = feas.shape[0]

            x_pred_norm = x_pred_norm.detach().numpy()

            x_pred_norm = x_pred_norm[feas==1]
            vector = z.detach().numpy()
            z = vector[feas==1]
            lof_score_val = lof_score_func(x_pred_norm)
            # concatenate arrays in the list
            x_pred['Class'] = cf_label.cpu().detach().numpy()
            x_true['Class'] = train_y.cpu().detach().numpy()
            x_true.to_csv(prefix_name+"_x_true.csv",index=False)
            
            x_pred.to_csv(prefix_name+"_x_pred.csv",index=False)

            z = pd.DataFrame(vector)
            z.to_csv(prefix_name+"_z_pred.csv",index=False)

            xprednorm = pd.DataFrame(x_pred_norm)
            xprednorm.to_csv(prefix_name+"_x_pred_norm.csv",index=False)

            print("----->> Current Average lof score {}".format(lof_score_val))
            #plot with tsne import tsne
            #lof = LocalOutlierFactor(n_neighbors=6, metric='euclidean')
            #lof_scores= lof.fit_predict(x_pred_norm) # tha mporouse kai gia x_true na ypologistei

            """from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=0)
            x_pred_norm_tsne = tsne.fit_transform(x_pred_norm)
            plt.scatter(x_pred_norm_tsne[:,0], x_pred_norm_tsne[:,1],c=lof_scores)
            plt.title('TSNE')
            plt.show()

            # calculate lof score
            k_lof_results=[]
            for k in range(3,500,3):
                lof = LocalOutlierFactor(n_neighbors=k, metric='euclidean')
                #lof.fit(x_pred2.drop(columns=['Class']))
                lof_scores= lof.fit_predict(x_pred_norm) # tha mporouse kai gia x_true na ypologistei
                #lof_scores = -lof.negative_outlier_factor_
                temp_lof_score = np.sum(lof_scores==-1)
                #lof_score_per_iter+=temp_lof_score
                k_lof_results.append(temp_lof_score)
            
            k_range = [*range(3,500,3)]
            print("k_lof_results", k_lof_results)
            #print(np.argmax(k_lof_results[0:100]))
            plt.plot(k_range, k_lof_results)
            plt.title('LOF CF')
            plt.xlabel('k')
            plt.ylabel('LOF')
            plt.show()"""

            full_lof.append(lof_score_val)
        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size
        test_size= test_size/sample_size
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))

    if case:
        pass
        """plt.plot(sample_range, valid_score_arr, '*', label='Val Age Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Age Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in Age')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()  """  
    print("Final average lof score", np.mean(np.array(full_lof)))
    return full_lof

# ------------------- MAIN FUNCTION FOR EVALUATION: CALLS THE FUNCTIONS FOR THE METRICS ------------------- 
def compute_eval_metrics_adult(immutables, methods, base_model_dir, encoded_size, pred_model, val_dataset, d, normalise_weights, mad_feature_weights, div_case, case, sample_range, filename,prefix_name='samples' ):   
    """
    Compute a specified evaluation metric for each trained FCX-VAE model on the Adult dataset.

    For each entry in `methods`, this function:
      1. Loads the corresponding VAE checkpoint.
      2. Samples counterfactuals on the held-out validation set.
      3. Computes a single metric given by `case`:
         - 0: validity (fraction of CFs that flip the classifier)
         - 1: feasibility (constraint score)
         - 2: continuous proximity
         - 3: categorical proximity
         - 4: LOF anomaly score
      4. Stores the result in a dictionary under the method’s key.

    Args:
        immutables (bool):
            If True, treats the last 4 features as immutable during CF generation.
        methods (dict[str, str]):
            Mapping from method name to VAE checkpoint filepath.
        base_model_dir (str):
            Directory where model checkpoints are stored.
        encoded_size (int):
            Latent dimensionality of the FCX‑VAE.
        pred_model (BlackBox):
            Pre-trained black-box classifier for validity checking.
        val_dataset (np.ndarray):
            Validation data array including labels, shape (N, d+1).
        d (DataLoader):
            DataLoader instance providing feature encodings and metadata.
        normalise_weights (dict[int, tuple[float, float]]):
            Per-feature (min, max) weights for proximity calculations.
        mad_feature_weights (dict[int, Any]):
            Feature weights used for LOF anomaly scoring.
        div_case (int):
            Diversity case identifier (unused in basic metrics).
        case (int):
            Metric case index to compute:
              0=validity, 1=feasibility, 2=cont-prox, 3=cat‑prox, 4=LOF.
        sample_range (list[int]):
            Indices of Monte Carlo samples to evaluate.
        filename (str):
            Base name for saving any output plots or arrays.
        prefix_name (str, optional):
            Prefix for naming sample output files (default: 'samples').

    Returns:
        dict:
            Mapping each method name to its computed metric value (float).
    """
    count=0
    fsize=20
    #fig = plt.figure(figsize=(7.7,6.5))
    final_res= {}
    dataset_name= 'adult'
    
    np.random.shuffle(val_dataset)
    x_sample= val_dataset[0,:]    
    x_sample= np.reshape( x_sample, (1, val_dataset.shape[1]))
    np.save('adult-visualise-sample.npy', x_sample)
        
    for key in methods.keys():

        #Loading torch model
        wm1=1e-2
        wm2=1e-2
        wm3=1e-2
        wm4=1e-2

        path= methods[key]
        cf_val=[]
            
        if immutables:
            fcx_vae = FCX_VAE( len(d.encoded_feature_names)-4, encoded_size, d,-4 )
        else:
            fcx_vae = FCX_VAE( len(d.encoded_feature_names), encoded_size, d )

        fcx_vae.load_state_dict(torch.load(path))
        fcx_vae.eval()
        learning_rate = 1e-2
        fcx_vae_optimizer = optim.Adam([
            {'params': filter(lambda p: p.requires_grad, fcx_vae.encoder_mean.parameters()),'weight_decay': wm1},
            {'params': filter(lambda p: p.requires_grad, fcx_vae.encoder_var.parameters()),'weight_decay': wm2},
            {'params': filter(lambda p: p.requires_grad, fcx_vae.decoder_mean.parameters()),'weight_decay': wm3}
        ], lr=learning_rate)        


        # Put the check for only Low to High Income CF
        train_x= torch.tensor( val_dataset ).float() 
        train_y = torch.argmax( pred_model(train_x), dim=1 ).numpy()
        val_dataset= val_dataset[ train_y==0 ]
            
        for i in range(10):
            if case==0:
                cf_val.append( validity_score(fcx_vae, pred_model, val_dataset, 0, sample_range,d) )
            elif case==1:
                val, inval= causal_score_age_constraint(fcx_vae, pred_model, val_dataset, d, normalise_weights, offset, 0, sample_range)
                cf_val.append( 100*np.array(val)/(np.array(val)+np.array(inval)) )
            elif case==2:
                val, inval= causal_score_age_ed_constraint(fcx_vae, pred_model, val_dataset, d, normalise_weights, offset, 0, sample_range)
                cf_val.append( 100*np.array(val)/(np.array(val)+np.array(inval)) )
            elif case==3:
                cf_val.append( proximity_score(fcx_vae, pred_model, val_dataset, d, mad_feature_weights, 0, 0, sample_range) )
            elif case==4:
                cf_val.append( proximity_score(fcx_vae, pred_model, val_dataset, d, mad_feature_weights, 1, 0, sample_range) )
            elif case==5:
                # Future work
                pass
            elif case==6:
                # Future work
                pass
            elif case==7:
                cf_val= causal_score_age_ed_constraint_lof(fcx_vae, pred_model, val_dataset, d, normalise_weights, offset, 0, sample_range,prefix_name=prefix_name)
            elif case==8:
                cf_val= causal_score_age_constraint_lof(fcx_vae, pred_model, val_dataset, d, normalise_weights, offset, 0, sample_range,prefix_name=prefix_name)

        final_res[key]= cf_val
        cf_val= np.mean( np.array(cf_val), axis=0 )
        if case==7 or case==8:
            cf_val=[cf_val]
        
        """if case==0:
            plt.title('Target Class Valid CF', fontsize=fsize)
            plt.xlabel('Total Counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of valid CF w.r.t. ML Classifier',  fontsize=fsize)
        elif case==1:
            plt.title('Constraint Valid CF: Age Constraint', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of CF satisfying Constraint', fontsize=fsize)
        elif case==2:
            plt.title('Constraint Valid CF: Age-Education Constraint', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of CF satisfying Constraint', fontsize=fsize)
        elif case==3:
            plt.title('Continuous Proximity Score', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Total change in continuous features', fontsize=fsize)
        elif case==4:
            plt.title('Categorical Proximity Score', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Total change in categorical features', fontsize=fsize)
        elif case==6:
            pass
        elif case==7 or case==8:
            plt.title('LOF', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Total lof', fontsize=fsize)"""

        if count==0:
            low = min(cf_val)
            high = max(cf_val)
        else:
            if low>min(cf_val):
                low=min(cf_val)
            elif high<max(cf_val):
                high=max(cf_val)
                
        """if case ==0 or case ==1 or case ==2:
            plt.ylim(0,101)
        elif case ==3 or case ==4:
            plt.ylim([np.ceil(low-0.5*np.abs(high)), np.ceil(high+0.5*np.abs(high))])
        else:
            plt.ylim([np.ceil(low-0.5-0.5*(high-low)), np.ceil(high+0.5+0.5*(high-low))])  """
        
        """if len(sample_range)==1:
            plt.plot(sample_range, cf_val, '.', label=key)
        else:
            plt.plot(sample_range, cf_val, label=key)  """          
            
        count+=1    
    
    #plt.legend(loc='lower left', fontsize=fsize/1.3)    

    os.makedirs('results/adult', exist_ok=True)
    #plt.savefig('results/adult/'+filename+'.jpg')
    #plt.show()
    return final_res

