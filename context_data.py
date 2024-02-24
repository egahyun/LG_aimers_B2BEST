import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

from sklearn.model_selection import train_test_split

cat_columns = [
    "customer_country",
    "business_subarea",
    "business_area",
    "business_unit",
    "customer_type",
    "customer_idx",
    "enterprise",
    "customer_job",
    "inquiry_type",
    "product_category",
    "product_subcategory",
    "product_modelname",
    "customer_position",
    "response_corporate",
    "expected_timeline",
    "category",
    "product_count",
    "timeline_count",
    "idit_all",
    "lead_owner",
    "bant_submit_count",
    "com_reg_count",
    "idx_count",
    "lead_count",
    "enterprise_count",
    "enterprise_weight"
]

def index_processing(context_df, train, test, column_name):
    idx = {v:k for k,v in enumerate(context_df[column_name].unique())}
    train.loc[:, column_name] = train[column_name].map(idx)
    test.loc[:, column_name] = test[column_name].map(idx)
    return idx

def process_context_data(train_df, test_df):
    context_df = pd.concat([train_df[cat_columns], test_df[cat_columns]]).reset_index(drop=True)
    idx = {}
    for col in cat_columns:
        idx_name = index_processing(context_df, train_df, test_df, col)
        idx[col+'2idx'] = idx_name
    return idx, train_df, test_df

def context_data_load():
    ######################## DATA LOAD
    train = pd.read_csv('train_last.csv', low_memory=False)
    test = pd.read_csv('submission_last.csv')

    # # 'customer_idx'와 'lead_owner' 카테고리별로 'is_converted'의 평균과 개수를 계산합니다.
    # ci_lo_mean = train.groupby(['customer_idx', 'lead_owner'])['is_converted'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)

    # # 변환율을 딕셔너리로 변환합니다.
    # conversion_dict = ci_lo_mean['mean'].to_dict()

    # # 'train' 데이터프레임에 새로운 열 'ci_lo_mean'을 추가하고 변환율을 매핑합니다.
    # train['ci_lo_mean'] = train.apply(lambda x: conversion_dict.get((x['customer_idx'], x['lead_owner']), np.nan), axis=1)
    # test['ci_lo_mean'] = test.apply(lambda x: conversion_dict.get((x['customer_idx'], x['lead_owner']), np.nan), axis=1)

    train = train.fillna(0)
    test = test.fillna(0)
    for column in train.drop(['is_converted'], axis=1).columns:
        if column in cat_columns:
            train[column] = train[column].astype('category')
            test[column] = test[column].astype('category')
        else:
            train[column] = train[column].astype('float')
            test[column] = test[column].astype('float')

    idx, context_train, context_test = process_context_data(train, test)
    field_dims = np.array([len(toidx) for toidx in idx], dtype=np.int32)

    data = {
            'train':context_train,
            'test':context_test,
            'field_dims':field_dims,
            'cat_columns' : cat_columns,
            }


    return data

def context_data_split(data, sampling_type: str = 'tomek', kfold: bool = True):
    X_resampled, y_resampled = 0, 0
    if sampling_type == 'smote':
        # SMOTE를 사용하여 데이터 오버샘플링
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(data['train'].drop(['is_converted'], axis=1), data['train']['is_converted'])
    elif sampling_type == 'adasyn':
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(data['train'].drop(['is_converted'], axis=1), data['train']['is_converted'])
    elif sampling_type == 'tomek':
        smoteto = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        X_resampled, y_resampled = smoteto.fit_resample(data['train'].drop(['is_converted'], axis=1), data['train']['is_converted'])

    data['X_samp'], data['y_samp'] = X_resampled, y_resampled

    if not kfold:
        # 샘플링된 데이터를 다시 훈련 데이터와 테스트 데이터로 분할
        X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, 
                                                        y_resampled, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y_resampled)

        y_train = y_train.astype(np.int32) ; y_valid = y_valid.astype(np.int32)
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data

def context_data_loader(data):
    """
    Parameters
    ----------
    batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    data_shuffle : bool
        data shuffle 여부
    ----------
    """
    batch_size = 1024
    data_shuffle = True

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data