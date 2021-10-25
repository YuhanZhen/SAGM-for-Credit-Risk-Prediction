#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on October, 2021
@author: ZihaoLi97
"""
import pandas as pd
import os
from lightgbm import LGBMClassifier
from data_preprocessing import data_loaded, feature_engineering
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_load_path', default='../../data/home-credit-default-risk-raw/',
                    help='the folder to save the training and test raw data')
parser.add_argument('--feature_combine', type=bool, default=False, help='whether to combine the intermediate vector '
                                                                        'from SAGNN as an auxiliary feature')
parser.add_argument('--model_choose', default='RF', choices=['LR', 'DT', 'RF', 'XGB', 'LGB'], help='choose machine learning method')
parser.add_argument('--valid_portion', type=float, default=0.1, help='the portion of validation part in train data')

opt, unknown = parser.parse_known_args()


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


def LR(train_df):
    train_index = []
    val_index = []
    for i in range(len(train_df)):
        if i % int(1 / opt.valid_portion) == 0:
            val_index.append(i)
        else:
            train_index.append(i)
    val_df = train_df.iloc[val_index]
    train_df = train_df.iloc[train_index]
    train_labels = train_df['TARGET']
    val_labels = val_df['TARGET']
    train_df.drop('SK_ID_CURR', inplace=True, axis=1)
    val_df.drop('SK_ID_CURR', inplace=True, axis=1)
    train_df.drop('TARGET', inplace=True, axis=1)
    val_df.drop('TARGET', inplace=True, axis=1)

    log_reg = LogisticRegression(C=0.0001)
    log_reg.fit(train_df, train_labels)
    log_reg_pred = log_reg.predict_proba(val_df)[:, 1]
    evaluation(val_labels, log_reg_pred)


def xgb(train_df, feature_train, feature_val):
    train_index = []
    val_index = []
    for i in range(len(train_df)):
        if i % int(1 / opt.valid_portion) == 0:
            val_index.append(i)
        else:
            train_index.append(i)
    val_df = train_df.iloc[val_index]
    train_df = train_df.iloc[train_index]
    train_labels = train_df['TARGET']
    val_labels = val_df['TARGET']
    train_df.drop('SK_ID_CURR', inplace=True, axis=1)
    val_df.drop('SK_ID_CURR', inplace=True, axis=1)
    train_df.drop('TARGET', inplace=True, axis=1)
    val_df.drop('TARGET', inplace=True, axis=1)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if opt.feature_combine:
        columns_feature = ['feature_nn'+str(i) for i in range(len(feature_train[0]))]
        train_feature_pd = pd.DataFrame(data=feature_train, columns=columns_feature)
        val_feature_pd = pd.DataFrame(data=feature_val, columns=columns_feature)
        train_df = pd.concat([train_df, train_feature_pd], axis=1)
        val_df = pd.concat([val_df, val_feature_pd], axis=1)

    ratio = (train_labels == 0).sum() / (train_labels == 1).sum()

    clf = XGBClassifier(n_estimators=10000, objective='binary:logistic', gamma=0.098, subsample=0.5, scale_pos_weight=ratio)
    # clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=100)
    clf.fit(train_df, train_labels, eval_set=[(val_df, val_labels)], eval_metric='auc', early_stopping_rounds=100)
    predictions = clf.predict_proba(val_df.values)[:, 1]
    evaluation(val_labels, predictions)


def lgb(train_df, feature_train, feature_val):
    train_index = []
    val_index = []
    for i in range(len(train_df)):
        if i % int(1 / opt.valid_portion) == 0:
            val_index.append(i)
        else:
            train_index.append(i)
    val_df = train_df.iloc[val_index]
    train_df = train_df.iloc[train_index]
    train_labels = train_df['TARGET']
    val_labels = val_df['TARGET']

    train_df.drop('SK_ID_CURR', inplace=True, axis=1)
    val_df.drop('SK_ID_CURR', inplace=True, axis=1)
    train_df.drop('TARGET', inplace=True, axis=1)
    val_df.drop('TARGET', inplace=True, axis=1)


    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if opt.feature_combine:
        columns_feature = ['feature_nn'+str(i) for i in range(len(feature_train[0]))]
        train_feature_pd = pd.DataFrame(data=feature_train, columns=columns_feature)
        val_feature_pd = pd.DataFrame(data=feature_val, columns=columns_feature)
        train_df = pd.concat([train_df, train_feature_pd], axis=1)
        val_df = pd.concat([val_df, val_feature_pd], axis=1)


    # clf = LGBMClassifier(
    #     nthread=4,
    #     n_estimators=10000,
    #     learning_rate=0.02,
    #     num_leaves=32,
    #     colsample_bytree=0.9497036,
    #     subsample=0.8715623,
    #     max_depth=8,
    #     reg_alpha=0.04,
    #     reg_lambda=0.073,
    #     min_split_gain=0.0222415,
    #     min_child_weight=40,
    #     silent=-1,
    #     verbose=-1)
    clf = LGBMClassifier(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=32,
        max_depth=8,
        min_child_weight=40,
        silent=-1,
        verbose=-1)
    clf.fit(train_df, train_labels, eval_metric='auc', eval_set=[(val_df, val_labels), (train_df, train_labels)],
              eval_names=['valid', 'train'], early_stopping_rounds=100, verbose=100)

    predictions = clf.predict_proba(val_df, axis=1)[:, 1]
    evaluation(val_labels, predictions)


def RF(train_df, feature_train, feature_val):
    train_index = []
    val_index = []
    for i in range(len(train_df)):
        if i % int(1 / 0.1) == 0:
            val_index.append(i)
        else:
            train_index.append(i)
    val_df = train_df.iloc[val_index]
    train_df = train_df.iloc[train_index]
    train_labels = train_df['TARGET']
    val_labels = val_df['TARGET']

    train_df.drop('SK_ID_CURR', inplace=True, axis=1)
    val_df.drop('SK_ID_CURR', inplace=True, axis=1)
    train_df.drop('TARGET', inplace=True, axis=1)
    val_df.drop('TARGET', inplace=True, axis=1)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if opt.feature_combine:
        columns_feature = ['feature_nn' + str(i) for i in range(len(feature_train[0]))]
        train_feature_pd = pd.DataFrame(data=feature_train, columns=columns_feature)
        val_feature_pd = pd.DataFrame(data=feature_val, columns=columns_feature)
        train_df = pd.concat([train_df, train_feature_pd], axis=1)
        val_df = pd.concat([val_df, val_feature_pd], axis=1)

    random_forest = RandomForestClassifier(n_estimators=1000, random_state=50, verbose=1, n_jobs=-1)
    random_forest.fit(train_df, train_labels)
    predictions = random_forest.predict_proba(val_df)[:, 1]
    evaluation(val_labels, predictions)


def DT(train_df):
    train_index = []
    val_index = []
    for i in range(len(train_df)):
        if i % int(1 / opt.valid_portion) == 0:
            val_index.append(i)
        else:
            train_index.append(i)
    val_df = train_df.iloc[val_index]
    train_df = train_df.iloc[train_index]
    train_labels = train_df['TARGET']
    val_labels = val_df['TARGET']
    train_df.drop('SK_ID_CURR', inplace=True, axis=1)
    val_df.drop('SK_ID_CURR', inplace=True, axis=1)
    train_df.drop('TARGET', inplace=True, axis=1)
    val_df.drop('TARGET', inplace=True, axis=1)

    decision_tree = DecisionTreeClassifier(max_depth=5)
    decision_tree = decision_tree.fit(train_df, train_labels)
    predictions = decision_tree.predict_proba(val_df)[:, 1]
    evaluation(val_labels, predictions)


def feature_load():
    with open(os.path.join(opt.data_load_path, 'train_features_intermediate.pkl'), 'rb') as f_train:
        train_feature = pickle.load(f_train)['feature_intermediate']
    with open(os.path.join(opt.data_load_path, 'val_features_intermediate.pkl'), 'rb') as f_val:
        val_feature = pickle.load(f_val)['feature_intermediate']
    return train_feature, val_feature


def main():
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = data_loaded()
    data_all = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    train_feature, val_feature = feature_load()
    if opt.model_choose == 'LR':
        LR(data_all)
    elif opt.model_choose == 'DT':
        DT(data_all)
    elif opt.model_choose == 'RF':
        RF(data_all, train_feature, val_feature)
    elif opt.model_choose == 'XGB':
        xgb(data_all, train_feature, val_feature)
    elif opt.model_choose == 'LGB':
        lgb(data_all, train_feature, val_feature)
    else:
        print('Error, the selected method must be one of LR, DT, RF, XGB, and LGB')


if __name__ == '__main__':
    main()
