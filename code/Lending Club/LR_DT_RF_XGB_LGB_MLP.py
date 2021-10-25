#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on October, 2021
@author: ZihaoLi97
"""
import pandas as pd
import os
import random
import numpy as np
import gc
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import pickle
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../../data/', help='the folder to save the raw data')
parser.add_argument('--sample', type=bool, default=True, help='whether to use sample data')
parser.add_argument('--SAGNN_feature', type=bool, default=False, help='whether to use SAGNN intermediate vector as auxiliary '
                                                               'feature for model (Randome Forest, XGBoost, LightGBM) training')
parser.add_argument('--split_portion', type=float, default=0.9, help='the portion of training data')
parser.add_argument('--models', type=str, default='LR', choices=['LR', 'DT', 'RF', 'XGB', 'LGB', 'MLP'],
                    help='choose to use what kinds of models')

opt, unknown = parser.parse_known_args()


def data_load(sample):
    if sample:
        accepted_data = pd.read_csv(os.path.join(opt.path, 'sample/lending_club_sample.csv'), low_memory=False,
                                    parse_dates=['issue_d'], infer_datetime_format=True, encoding='utf_8_sig')
    else:
        accepted_data = pd.read_csv(os.path.join(opt.path, 'lending_club/accepted_2007_to_2018Q4.csv'),
                                    low_memory=False,
                                    parse_dates=['issue_d'], infer_datetime_format=True, encoding='utf_8_sig')
    return accepted_data


def feature_load():
    with open(os.path.join(opt.path, 'train_features_intermediate.pkl'), 'rb') as handle:
        train_feature = list(pickle.load(handle).values())[0]

    with open(os.path.join(opt.path, 'val_features_intermediate.pkl'), 'rb') as handle:
        test_feature = list(pickle.load(handle).values())[0]
    return train_feature, test_feature


def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])


def data_preprocessing(accepted_data_raw, ratio_train=0.9):
    accepted_data_raw = accepted_data_raw.loc[accepted_data_raw['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    missing_fractions = accepted_data_raw.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    accepted_data_raw.drop(labels=drop_list, axis=1, inplace=True)
    keep_list_kd = ['addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length',
                    'fico_range_high', 'fico_range_low', 'home_ownership', 'initial_list_status',
                    'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec',
                    'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'total_acc',
                    'verification_status']

    accepted_data = accepted_data_raw[keep_list_kd]

    accepted_data['term'] = accepted_data['term'].apply(lambda s: np.int8(s.split()[0])).value_counts(normalize=True)
    accepted_data['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
    accepted_data['emp_length'].replace('< 1 year', '0 years', inplace=True)
    accepted_data['emp_length'] = accepted_data['emp_length'].apply(emp_length_to_int)
    accepted_data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
    accepted_data['log_annual_inc'] = accepted_data['annual_inc'].apply(lambda x: np.log10(x + 1))
    accepted_data.drop('annual_inc', axis=1, inplace=True)
    accepted_data['earliest_cr_line'] = accepted_data['earliest_cr_line'].apply(lambda s: int(s[-4:]))
    accepted_data['fico_score'] = 0.5 * accepted_data['fico_range_low'] + 0.5 * accepted_data['fico_range_high']
    accepted_data.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)
    accepted_data['log_revol_bal'] = accepted_data['revol_bal'].apply(lambda x: np.log10(x + 1))
    accepted_data.drop('revol_bal', axis=1, inplace=True)
    accepted_data['loan_label'] = (accepted_data['loan_status'] == 'Charged Off').apply(np.uint8)
    accepted_data.drop('loan_status', axis=1, inplace=True)

    object_column_list = []
    for column_temp in list(accepted_data.columns):
        if accepted_data[column_temp].dtype == 'object':
            object_column_list.append(column_temp)
    accepted_data = pd.get_dummies(accepted_data, columns=object_column_list, drop_first=True)

    columns_list = list(accepted_data.columns)
    columns_list.remove('loan_label')
    columns_list.remove('issue_d')
    accepted_data[columns_list] = (accepted_data[columns_list] - accepted_data[columns_list].min()) / \
                               (accepted_data[columns_list].max() - accepted_data[columns_list].min())

    accepted_data['issue_d'] = pd.to_datetime(accepted_data['issue_d'])
    accepted_data = accepted_data.fillna(0)
    data_train = accepted_data.loc[accepted_data['issue_d'] < accepted_data['issue_d'].quantile(ratio_train)]
    data_test = accepted_data.loc[accepted_data['issue_d'] >= accepted_data['issue_d'].quantile(ratio_train)]

    data_train.drop('issue_d', axis=1, inplace=True)
    data_test.drop('issue_d', axis=1, inplace=True)


    del accepted_data
    gc.collect()
    return data_train, data_test


def feature_combine(data_train, data_test, feature_train, feature_test):

    df_train_feature = pd.DataFrame(data=feature_train, columns=['feature_'+str(i) for i in range(len(feature_train[0]))])
    df_test_feature = pd.DataFrame(data=feature_test,
                                    columns=['feature_' + str(i) for i in range(len(feature_train[0]))])
    data_train = pd.concat([data_train, df_train_feature], axis=1)
    data_test = pd.concat([data_test, df_test_feature], axis=1)
    data_train = data_train.fillna(0)
    data_test = data_test.fillna(0)
    return data_train, data_test


def evaluation(groundtruth, predicted):
  predict_label = []
  for i in predicted:
    if i > 0.5:
      predict_label.append(1)
    else:
      predict_label.append(0)
  print('Val precision: %.5f'%precision_score(groundtruth, predict_label, average='macro'))
  print('Val recall: %.5f'%recall_score(groundtruth, predict_label, average='macro'))
  print('Val F1 %.5f'%f1_score(groundtruth, predict_label, average='macro'))
  fpr, tpr, thresholds = roc_curve(groundtruth, predicted, pos_label=1)
  print('Val AUC: %.5f'%auc(fpr, tpr))


def logistic(train_data, test_data):
    y_train = train_data['loan_label']
    X_train = train_data.drop(['loan_label'], axis=1)
    y_test = test_data['loan_label']
    X_test = test_data.drop(['loan_label'], axis=1)
    del train_data
    gc.collect()

    log_reg = LogisticRegression(C=0.0001)
    log_reg.fit(X_train, y_train)

    log_reg_pred = log_reg.predict_proba(X_test)[:, 1]
    evaluation(y_test, log_reg_pred)


def DT(train_data, test_data):
    y_train = train_data['loan_label']
    X_train = train_data.drop(['loan_label'], axis=1)
    y_test = test_data['loan_label']
    X_test = test_data.drop(['loan_label'], axis=1)
    del train_data
    gc.collect()

    decision_tree = DecisionTreeClassifier(max_depth=5)
    decision_tree = decision_tree.fit(X_train, y_train)
    predictions = decision_tree.predict_proba(X_test)[:, 1]
    evaluation(y_test, predictions)


def randomforest(train_data, test_data):
    y_train = train_data['loan_label']
    X_train = train_data.drop(['loan_label'], axis=1)
    y_test = test_data['loan_label']
    X_test = test_data.drop(['loan_label'], axis=1)
    del train_data
    gc.collect()

    random_forest = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    predictions = random_forest.predict_proba(X_test)[:, 1]
    evaluation(y_test, predictions)


def XGB(train_data, test_data):
    y_train = train_data['loan_label']
    X_train = train_data.drop(['loan_label'], axis=1)
    y_test = test_data['loan_label']
    X_test = test_data.drop(['loan_label'], axis=1)
    del train_data
    gc.collect()

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    clf = XGBClassifier(n_estimators=1200, objective='binary:logistic', gamma=0.098, subsample=0.5,
                        scale_pos_weight=ratio)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=10)
    predictions = clf.predict_proba(X_test.values)[:, 1]
    evaluation(y_test, predictions)


def lgb(train_data, test_data):
    y_train = train_data['loan_label']
    X_train = train_data.drop(['loan_label'], axis=1)
    y_test = test_data['loan_label']
    X_test = test_data.drop(['loan_label'], axis=1)
    del train_data
    gc.collect()

    clf = LGBMClassifier(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=32,
        max_depth=8,
        min_child_weight=40,
        silent=-1,
        verbose=-1)

    clf.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_test, y_test), (X_train, y_train)],
              eval_names=['valid', 'train'], early_stopping_rounds=100, verbose=100)

    predictions = clf.predict_proba(X_test, axis=1)[:, 1]
    evaluation(y_test, predictions)


def data_preprocessing_mlp(accepted_data, ratio_train=0.9):
    accepted_data = accepted_data.loc[accepted_data['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    missing_fractions = accepted_data.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    accepted_data.drop(labels=drop_list, axis=1, inplace=True)
    keep_list_kd = ['addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length',
                    'fico_range_high', 'fico_range_low', 'home_ownership', 'initial_list_status',
                    'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec',
                    'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'total_acc',
                    'verification_status']

    accepted_data = accepted_data[keep_list_kd]

    accepted_data['term'] = accepted_data['term'].apply(lambda s: np.int8(s.split()[0])).value_counts(normalize=True)
    accepted_data['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
    accepted_data['emp_length'].replace('< 1 year', '0 years', inplace=True)
    accepted_data['emp_length'] = accepted_data['emp_length'].apply(emp_length_to_int)
    accepted_data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
    accepted_data['log_annual_inc'] = accepted_data['annual_inc'].apply(lambda x: np.log10(x + 1))
    accepted_data.drop('annual_inc', axis=1, inplace=True)
    accepted_data['earliest_cr_line'] = accepted_data['earliest_cr_line'].apply(lambda s: int(s[-4:]))
    accepted_data['fico_score'] = 0.5 * accepted_data['fico_range_low'] + 0.5 * accepted_data['fico_range_high']
    accepted_data.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)
    accepted_data['log_revol_bal'] = accepted_data['revol_bal'].apply(lambda x: np.log10(x + 1))
    accepted_data.drop('revol_bal', axis=1, inplace=True)
    accepted_data['loan_label'] = (accepted_data['loan_status'] == 'Charged Off').apply(np.uint8)
    accepted_data.drop('loan_status', axis=1, inplace=True)

    object_column_list = []
    for column_temp in list(accepted_data.columns):
        if accepted_data[column_temp].dtype == 'object':
            accepted_data[column_temp] = accepted_data[column_temp].astype('category').cat.codes
            object_column_list.append(column_temp)

    # accepted_data = pd.get_dummies(accepted_data, columns=object_column_list, drop_first=True)

    columns_list = list(accepted_data.columns)
    columns_list.remove('loan_label')
    columns_list.remove('issue_d')
    accepted_data[columns_list] = (accepted_data[columns_list] - accepted_data[columns_list].min()) / \
                               (accepted_data[columns_list].max() - accepted_data[columns_list].min())

    accepted_data['issue_d'] = pd.to_datetime(accepted_data['issue_d'])
    accepted_data = accepted_data.fillna(0)
    data_train = accepted_data.loc[accepted_data['issue_d'] < accepted_data['issue_d'].quantile(ratio_train)]
    data_test = accepted_data.loc[accepted_data['issue_d'] >= accepted_data['issue_d'].quantile(ratio_train)]

    data_train.drop('issue_d', axis=1, inplace=True)
    data_test.drop('issue_d', axis=1, inplace=True)

    del accepted_data
    gc.collect()
    return data_train, data_test


def auc_calculate(groundtruth, predicted_prob):
    fpr, tpr, thresholds = roc_curve(groundtruth, predicted_prob, pos_label=1)
    return auc(fpr, tpr)


class nn_mlp(nn.Module):
    def __init__(self, hidden_size):
        super(nn_mlp, self).__init__()

        self.layer_1 = nn.Linear(hidden_size, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.layer_3 = nn.Linear(32, 32)
        self.layer_4 = nn.Linear(32, 16)
        self.layer_5 = nn.Linear(16, 2)
        # self.dropout_layer = nn.Dropout(0.15)

        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, data_temp):
        output = self.layer_norm_1(data_temp)
        output = F.relu(self.layer_1(output))
        output = F.relu(self.layer_2(output))
        output = F.relu(self.layer_3(output))
        output = F.relu(self.layer_4(output))
        output = F.relu(self.layer_5(output))

        return output


def mlp(data_train, data_test):
    y_train = data_train['loan_label']
    X_train = data_train.drop(['loan_label'], axis=1)
    y_test = data_test['loan_label']
    X_test = data_test.drop(['loan_label'], axis=1)
    del data_train, data_test
    gc.collect()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_value = torch.tensor(np.array(X_train)).float()
    train_label = torch.tensor(np.array(y_train)).long()
    val_value = torch.tensor(np.array(X_test)).float()
    val_label = torch.tensor(np.array(y_test)).long()

    train_dataset = Data.TensorDataset(train_value, train_label)
    loader_train = Data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=500)
    val_dataset = Data.TensorDataset(val_value, val_label)
    loader_val = Data.DataLoader(dataset=val_dataset, shuffle=False, batch_size=500)

    nn_model = nn_mlp(hidden_size=len(X_train.columns)).to(device)
    # optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=0.1)
    optimizer = optim.SGD(nn_model.parameters(), lr=0.01)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        nn_model.cuda()
    epoch_num = 200

    val_auc_best = 0
    count = 0

    print('Training start')
    for epoch_temp in range(epoch_num):
        loss = 0
        auc_train = []
        for i in loader_train:

            nn_model.train()
            optimizer.zero_grad()
            output = nn_model(i[0].to(device))
            output_copy = output
            outputs_predit = list(F.softmax(output_copy, dim=-1).cpu().detach().numpy()[:, 1])
            auc_train.append(auc_calculate(i[1].numpy(), outputs_predit))
            loss = criterion(output.squeeze(), i[1].to(device))
            loss.backward()
            optimizer.step()
            loss += loss.item()
        loss = loss / len(loader_train)
        print('Loss in Epoch {0}: {1}'.format(epoch_temp, loss))
        print('Train AUC in Epoch {0}: {1}'.format(epoch_temp, np.mean(auc_train)))

        count += 1
        auc_val = []
        with torch.no_grad():
            for batch_data in loader_val:
                nn_model.eval()
                optimizer.zero_grad()
                output = nn_model(batch_data[0].to(device))
                outputs_predit = list(F.softmax(output, dim=-1).cpu().numpy()[:, 1])
                auc_val.append(auc_calculate(batch_data[1].numpy(), outputs_predit))
            print('Val AUC in Epoch {0}: {1}'.format(epoch_temp, np.mean(auc_val)))

        if np.mean(auc_val) > val_auc_best:
            val_auc_best = np.mean(auc_val)
            print('Best Val AUC in Epoch {0}: {1}'.format(epoch_temp, val_auc_best))
            best_model = nn_model
            count = 0

        if count > 10:
            print('Early stop')
            break
        schedule.step()
    print('Best Val AUC: {0}'.format(val_auc_best))
    return val_auc_best


def main():
    accepted_data = data_load(sample=opt.sample)
    data_train, data_test = data_preprocessing(accepted_data, ratio_train=opt.split_portion)
    data_train_mlp, data_test_mlp = data_preprocessing_mlp(accepted_data, ratio_train=opt.split_portion)
    if opt.SAGNN_feature:
        feature_train, feature_test = feature_load()
        data_train, data_test = feature_combine(data_train, data_test, feature_train, feature_test)
    if opt.models == 'LR':
        logistic(data_train, data_test)
    elif opt.models == 'DT':
        DT(data_train, data_test)
    elif opt.models == 'RF':
        randomforest(data_train, data_test)
    elif opt.models == 'XGB':
        XGB(data_train, data_test)
    elif opt.models == 'LGB':
        lgb(data_train, data_test)
    elif opt.models == 'mlp':
        mlp(data_train_mlp, data_test_mlp)
    else:
        print('Error: the model must be any of LR, DT, RF, XGB, LGB, mlp.')


if __name__ == '__main__':
    main()
