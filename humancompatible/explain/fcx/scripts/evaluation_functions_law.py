from scripts.fcx_vae_model import FCX_VAE
import sys,os
import random
import pandas as pd
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
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
from sklearn.neighbors import LocalOutlierFactor
#Seed for repoduability
torch.manual_seed(10000000)

offset=0.0

# law school both lsat and tier
encoded_feature_names=['lsat', 'decile1b', 'decile3', 'ugpa', 'zfygpa', 'zgpa', 'fulltime_1', 'fulltime_2',  'fam_inc_1', 'fam_inc_2', 'fam_inc_3', 'fam_inc_4', 'fam_inc_5', 'tier_1', 'tier_2', 'tier_3', 'tier_4', 'tier_5', 'tier_6','male_0', 'male_1']


education_score= {'1': 6,
        '2': 5,
        '3': 4,
        '4': 3,
        '5': 2,
        '6': 1}

ed_dict={}
for key in education_score.keys():
    ed_dict[encoded_feature_names.index('tier_'+key)]= education_score[key]
print(ed_dict)

# ------------------- HELPER FUNCTIONS ------------------- 
def de_normalise( x, normalise_weights ):
    return (normalise_weights[1] - normalise_weights[0])*x + normalise_weights[0]

def de_scale( x, normalise_weights ):
    return (normalise_weights[1] - normalise_weights[0])*x

# ------------------- VALIDITY METRIC ------------------- 
def validity_score(model, pred_model, train_dataset, case, sample_range,d=None):

    x_pred_list = []
    validity_score_arr=[]
    vector = np.array([])
    x_pred_full = pd.DataFrame()
    x_true_full = pd.DataFrame()
    z_pred_full = pd.DataFrame()
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y = torch.argmax( pred_model(train_x), dim=1 )                
        valid_cf_count=0        
        for sample_iter in range(sample_size):
               
            recon_err, kl_err, x_true, x_pred, cf_label,z = model.compute_elbo( train_x, 1.0-train_y, pred_model,True )
            #add x train and x pred to dataframe
            if d!=None:
                x_pred2= d.de_normalize_data( d.get_decoded_data(x_pred.cpu().detach().numpy()) )
                x_true2= d.de_normalize_data( d.get_decoded_data(x_true.cpu().detach().numpy()) )                
            
            x_pred2['income'] = cf_label.cpu().detach().numpy()
            x_true2['income'] = train_y.cpu().detach().numpy()
            vector = z.cpu().detach().numpy()
            
            z = pd.DataFrame(vector)
            x_pred_full = x_pred_full.append(x_pred2)
            x_true_full = x_true_full.append(x_true2)
            z_pred_full = z_pred_full.append(z)
            
            cf_label = cf_label.numpy()
            valid_cf_count += np.sum( train_y.numpy() != cf_label )
            
        test_size= train_x.shape[0]
        valid_cf_count=valid_cf_count/sample_size
        validity_score_arr.append(100*valid_cf_count/test_size)
    
    if case:
        plt.plot(sample_range, validity_score_arr)
        plt.title('Valid CF')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()
    print('Mean Validity Score: ', np.mean(np.array(validity_score_arr)) )
    return validity_score_arr

# ------------------- PROXIMITY METRIC ------------------- 
def proximity_score(model, pred_model, train_dataset, d, mad_feature_weights, cat, case, sample_range):
    

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
        plt.plot(sample_range, prox_score_arr)
        if cat:
            plt.title('Categorical Proximity')
        else:
            plt.title('Continuous Proximity')

        plt.xlabel('Sample Size')
        plt.ylabel('Magnitude')
        plt.show()
    print('Mean Proximity Score: ', np.mean(np.array(prox_score_arr)) )
    return prox_score_arr


def lof_score_func(data):
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
def causal_score_age_ed_constraint_lof(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range,prefix_name):
        
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

        full_lof=[]
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label,z= model.compute_elbo( train_x, 1.0-train_y, pred_model ,True)            
            x_pred_norm = x_pred.clone()  
            
            feas = []
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            ed_idx = x_true.columns.get_loc('tier')
            age_idx = x_true.columns.get_loc('lsat')            

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
            try:
                lof_score_val = lof_score_func(x_pred_norm)
            except Exception as e:
                print("There are not enough valid CF examples for LOF calculation. Please increase Validity")
                exit(-1)
            # concatenate arrays in the list
            x_pred['Class'] = cf_label.cpu().detach().numpy()
            x_true['Class'] = train_y.cpu().detach().numpy()
            x_true.to_csv(prefix_name+"_x_true.csv",index=False)
            x_pred.to_csv(prefix_name+"_x_pred.csv",index=False)
            z = pd.DataFrame(vector)
            z.to_csv(prefix_name+"_z_pred.csv",index=False)
            xprednorm = pd.DataFrame(x_pred_norm)
            xprednorm.to_csv(prefix_name+"_x_pred_norm.csv",index=False)
            print("----->> Average lof score {}".format(lof_score_val))

            full_lof.append(lof_score_val)
        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size
        
        test_size= test_size/sample_size
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))
    full_lof_score = np.mean(np.array(full_lof))

    return full_lof

# ------------------- BINARY CONSTRAINT ------------------- 
def causal_score_age_ed_constraint(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range):
        
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
            ed_idx = x_true.columns.get_loc('tier')
            age_idx = x_true.columns.get_loc('lsat')            

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
        plt.plot(sample_range, valid_score_arr, '*', label='Val Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('All Education Levels')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()
    print('Mean Age-Ed Constraint Score: ', valid_score, invalid_score, valid_score/(valid_score+invalid_score))
    print('Count: ', count1, count2, count3, count1+count2+count3)
    print('Pos Count: ', pos1, pos2, pos3 )
    if count1 and count2 and count3:
        print('Pos Percentage: ', pos1/count1, pos2/count2, pos3/count3 )    
    return valid_score_arr, invalid_score_arr


# ------------------- UNARY CONSTRAINT ------------------- 
def causal_score_age_constraint(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range):
        
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
            age_idx = x_true.columns.get_loc('lsat')            
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
        plt.plot(sample_range, valid_score_arr, '*', label='Val Age Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Age Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in Age')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()    
    print('Mean Age Constraint Score: ', valid_score, invalid_score, valid_score/(valid_score+invalid_score))
    return valid_score_arr, invalid_score_arr

# ------------------- UNARY CONSTRAINT + LOF ------------------- 
def causal_score_age_constraint_lof(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range,prefix_name):
        
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

            age_idx = x_true.columns.get_loc('lsat')            
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
            try:
                lof_score_val = lof_score_func(x_pred_norm)
            except Exception as e:
                print("There are not enough valid CF examples for LOF calculation. Please increase Validity")
                exit(-1)
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
            
            full_lof.append(lof_score_val)
        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size

        test_size= test_size/sample_size
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))

    return full_lof


#------------------- MAIN FUNCTION FOR EVALUATION: CALLS THE FUNCTIONS FOR THE METRICS ------------------- 

def compute_eval_metrics(immutables, methods, base_model_dir, encoded_size, pred_model, val_dataset, d, normalise_weights, mad_feature_weights, div_case, case, sample_range, filename,prefix_name='test'):   
    count=0
    fsize=20
    fig = plt.figure(figsize=(7.7,6.5))
    final_res= {}
    dataset_name= 'law'
    
    np.random.shuffle(val_dataset)
    x_sample= val_dataset[0,:]    
    x_sample= np.reshape( x_sample, (1, val_dataset.shape[1]))
    np.save('law-visualise-sample.npy', x_sample)  
    
    for key in methods.keys():

        wm1=1e-2
        wm2=1e-2
        wm3=1e-2
        wm4=1e-2

        path= methods[key]
        cf_val=[]
            
        if immutables:
            fcx_vae = FCX_VAE( len(d.encoded_feature_names)-2, encoded_size, d,-2 )
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
                cf_val= causal_score_age_ed_constraint_lof(fcx_vae, pred_model, val_dataset, d, normalise_weights, offset, 0, sample_range,prefix_name)
            elif case==8:
                cf_val= causal_score_age_constraint_lof(fcx_vae, pred_model, val_dataset, d, normalise_weights, offset, 0, sample_range,prefix_name)

        final_res[key]= cf_val
        cf_val= np.mean( np.array(cf_val), axis=0 )

        if case==7 or case==8:
            cf_val=[cf_val]
        
        if case==0:
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
            plt.ylabel('Total lof', fontsize=fsize)
        
        if count==0:
            low = min(cf_val)
            high = max(cf_val)
        else:
            if low>min(cf_val):
                low=min(cf_val)
            elif high<max(cf_val):
                high=max(cf_val)
                
        if case ==0 or case ==1 or case ==2:
            plt.ylim(0,101)
        elif case ==3 or case ==4:
            plt.ylim([np.ceil(low-0.5*np.abs(high)), np.ceil(high+0.5*np.abs(high))])
        else:
            plt.ylim([np.ceil(low-0.5-0.5*(high-low))-0.2, np.ceil(high+0.5+0.5*(high-low))])  
        
        if len(sample_range)==1:
            plt.plot(sample_range, cf_val, '.', label=key)
        else:
            plt.plot(sample_range, cf_val, label=key)            
            
        count+=1    
    
    plt.legend(loc='lower left', fontsize=fsize/1.3)    
    os.makedirs('results/law', exist_ok=True)
    plt.savefig('results/law/'+filename+'.jpg')
    plt.show()
    print(cf_val)
    return final_res