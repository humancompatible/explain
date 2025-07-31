from dataloader import DataLoader
from helpers import *
import sys
import pandas as pd
import numpy as np
import json

base_dir='../data/'
dataset_name=sys.argv[1]

#Dataset
if dataset_name=='adult':
    dataset = load_adult_income_dataset()

    params= {'dataframe':dataset.copy(), 'continuous_features':['age','hours_per_week'], 'outcome_name':'income'}
    d = DataLoader(params)
    train_data_vae= d.data_df.copy()

    train_data_vae_l0= train_data_vae[ train_data_vae['income']==0 ]
    train_data_vae_l0= train_data_vae_l0[ train_data_vae_l0['age'] > 35 ]
    train_data_vae_l1= train_data_vae[ train_data_vae['income']==1 ]
    train_data_vae_l1= train_data_vae_l1[ train_data_vae_l1['age'] < 45 ]
    train_data_vae= pd.concat( [ train_data_vae_l0, train_data_vae_l1 ], axis=0 )
    
    columns= train_data_vae.columns

if dataset_name=='folktables_adult':
    dataset = load_adult_income_dataset_folktables()

    params= {'dataframe':dataset.copy(), 'continuous_features':['age','hours_per_week'], 'outcome_name':'income'}
    d = DataLoader(params)
    train_data_vae= d.data_df.copy()

    train_data_vae_l0= train_data_vae[ train_data_vae['income']==0 ]
    train_data_vae_l0= train_data_vae_l0[ train_data_vae_l0['age'] > 35 ]
    train_data_vae_l1= train_data_vae[ train_data_vae['income']==1 ]
    train_data_vae_l1= train_data_vae_l1[ train_data_vae_l1['age'] < 45 ]
    train_data_vae= pd.concat( [ train_data_vae_l0, train_data_vae_l1 ], axis=0 )
    
    columns= train_data_vae.columns

if dataset_name=='census':
    dataset = pd.read_csv("../data/census/census_data.csv")

    params= {'dataframe':dataset.copy(), 'continuous_features':['age','wage_per_hour','capital_gains','capital_losses','dividends_from_stocks','num_persons_worked_for_employer','weeks_worked_in_year'], 'outcome_name':'income'}
    d = DataLoader(params)
    train_data_vae= d.data_df.copy()
    train_data_vae_l0= train_data_vae[ train_data_vae['income']==0 ]
    train_data_vae_l1= train_data_vae[ train_data_vae['income']==1 ]
    train_data_vae= pd.concat( [ train_data_vae_l0, train_data_vae_l1 ], axis=0 )
    
    columns= train_data_vae.columns

if dataset_name=='law':
    dataset = pd.read_csv("../data/bar_pass_pred/bar_pass_prediction_v2.csv")
    params= {'dataframe':dataset.copy(), 'continuous_features':['decile1b','decile3', 'lsat','ugpa','zfygpa','zgpa'], 'outcome_name':'pass_bar'}
    d = DataLoader(params)
    train_data_vae= d.data_df.copy()
    train_data_vae_l0= train_data_vae[ train_data_vae['pass_bar']==0 ]
    train_data_vae_l1= train_data_vae[ train_data_vae['pass_bar']==1 ]
    train_data_vae= pd.concat( [ train_data_vae_l0, train_data_vae_l1 ], axis=0 )
    
    columns= train_data_vae.columns



#MAD
mad_feature_weights = d.get_mads_from_training_data(normalized=False)
print(mad_feature_weights)

#One Hot Encoding for categorical features
encoded_data = d.one_hot_encode_data(train_data_vae)
dataset = encoded_data.to_numpy()

#Normlaise_Weights
data_size = len(d.encoded_feature_names)
encoded_categorical_feature_indexes = d.get_data_params()[2]     
encoded_continuous_feature_indexes=[]
for i in range(data_size):
    valid=1
    for v in encoded_categorical_feature_indexes:
        if i in v:
            valid=0
    if valid:
        encoded_continuous_feature_indexes.append(i)            
encoded_start_cat = len(encoded_continuous_feature_indexes)
normalise_weights={}
for idx in encoded_continuous_feature_indexes:
    _max= float(np.max( dataset[:,idx] ))
    _min= float(np.min( dataset[:,idx] ))
    normalise_weights[idx]=[_min, _max]

#Normlization for conitnuous features
encoded_data= d.normalize_data(encoded_data)

if dataset_name=='census':
    print("edw")
    print(encoded_data.columns) 

if dataset_name=='adult':
    # Need to rearrange columns such that the Income comes at the last
    cols = list(encoded_data.columns)
    cols = cols[:2] + cols[3:] + [cols[2]]
    encoded_data = encoded_data[cols]

if dataset_name=='folktables_adult':
    # Need to rearrange columns such that the Income comes at the last
    cols = list(encoded_data.columns)
    #cols = cols[:2] + cols[3:] + [cols[2]]
    cols.remove('income')
    cols.append('income')
    encoded_data = encoded_data[cols]
    
if dataset_name=='law':
    #data = pd.read_csv("../data/bar_pass_pred/bar_pass_prediction_v2.csv")
    cols = list(encoded_data.columns)
    cols = [cols[5]] + cols[:5] + cols[7:] + [cols[6]]
    
    encoded_data = encoded_data[cols]


dataset = encoded_data.to_numpy()

#Train, Val, Test Splits
np.random.shuffle(dataset)
test_size= int(0.1*dataset.shape[0])
vae_test_dataset= dataset[:test_size]
dataset= dataset[test_size:]
vae_val_dataset= dataset[:test_size]
vae_train_dataset= dataset[test_size:]

# Saving dataets 
np.save(base_dir+dataset_name+'-'+'train-set', vae_train_dataset )
np.save(base_dir+dataset_name+'-'+'val-set', vae_val_dataset )
np.save(base_dir+dataset_name+'-'+'test-set', vae_test_dataset )

#Saving Normalise Weights
f=open(base_dir+dataset_name+'-'+'normalise_weights.json', 'w')
json.dump(normalise_weights, f)
f.close()

#Saving MAD 
f=open(base_dir+dataset_name+'-'+'mad.json', 'w')
json.dump(mad_feature_weights, f)
f.close()
