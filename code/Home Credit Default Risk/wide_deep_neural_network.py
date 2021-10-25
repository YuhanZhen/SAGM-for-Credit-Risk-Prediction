#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on October, 2021
@author: ZihaoLi97
"""
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.utils.data as Data
from data_preprocessing import data_loaded, feature_engineering
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import warnings
import argparse

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--valid_portion', type=float, default=0.1, help='the portion of validation part in train data')
parser.add_argument('--optimizer', default=optim.Adam, help='optimizer for training')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--step_size', type=float, default=50, help='the interval epoch to decay learning rate')
parser.add_argument('--EmbeddingSize', type=int, default=5, help='Embedding size for enumerated variable')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0, help='learning rate decay rate')
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')

opt, unknown = parser.parse_known_args()


class CLASS_Wide_Deep(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_num = opt.EmbeddingSize
        self.embedding_column = nn.Embedding(kwargs['bag_size'], self.embedding_num)
        self.deep_layer_1 = nn.Linear(opt.EmbeddingSize * kwargs['num_embed'] + kwargs['num_val'], 256)
        self.deep_layer_2 = nn.Linear(256, 128)
        self.deep_layer_3 = nn.Linear(128, 64)
        self.deep_layer_4 = nn.Linear(64, 1)
        # self.deep_layer_5 = nn.Linear(32, 2)
        self.wide_layer = nn.Linear(kwargs['wide_num'], 1)
        self.output_layer = nn.Linear(1+1, 1)
        self.dropout_cat = nn.Dropout(0.35)

    def forward(self, value_batch, embedd_batch, wide_batch):
        embedd_batch = self.embedding_column(embedd_batch)
        embedd_layer = embedd_batch.flatten(-2)
        self.concat_deep = torch.cat((embedd_layer, value_batch), 1)

        self.cat_1 = self.dropout_cat(F.relu(self.deep_layer_1(self.concat_deep)))

        self.cat_2 = self.dropout_cat(F.relu(self.deep_layer_2(self.cat_1)))
        self.cat_3 = self.dropout_cat(F.relu(self.deep_layer_3(self.cat_2)))
        self.cat_4 = self.dropout_cat(F.relu(self.deep_layer_4(self.cat_3)))
        # self.cat_5 = self.dropout_cat(F.relu(self.deep_layer_5(self.cat_4)))

        self.wide = F.sigmoid(self.wide_layer(wide_batch))
        self.output = F.sigmoid(self.output_layer(torch.cat((self.cat_4, self.wide), 1)))

        return self.output


def containsAny(seq, aset):
    return True if any(i in seq for i in aset) else False


def data_preprocessed(data_all):
    list_columns = data_all.columns.tolist()
    list_columns.remove('SK_ID_CURR')
    list_columns.remove('TARGET')

    columns_value_selected = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE',
                              'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
                              'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE', 'index']
    value_column = ['MEAN', 'SUM', 'MIN', 'MAX', 'COUNT', 'RATIO', 'STD', 'AVG', 'TOTAL', 'MODE', 'MEDI']
    columns_embedd_selected = []
    wide_df_dict = {}
    for column_temp in list_columns:
        if containsAny(column_temp, value_column):
            columns_value_selected.append(column_temp)
        elif len(data_all[column_temp].unique()) > 50:
            columns_value_selected.append(column_temp)
        else:
            columns_embedd_selected.append(column_temp)
    data_all = data_all.fillna(0)
    for column_temp in list_columns:
        if len(data_all[column_temp].unique()) <= 20:
            temp = pd.get_dummies(data_all[column_temp])
            for temp_name in temp.columns.tolist():
                wide_df_dict[column_temp + '_' + str(temp_name)] = list(temp[temp_name])

    data_all[list(wide_df_dict.keys())] = wide_df_dict.values()
    data_all[columns_value_selected] = (data_all[columns_value_selected] - data_all[columns_value_selected].min()) / \
                                       (data_all[columns_value_selected].max() - data_all[columns_value_selected].min())

    start_encode = 0
    for column_temp in columns_embedd_selected:
        dict_temp = {}
        for i in data_all[column_temp].unique():
            dict_temp[i] = start_encode
            start_encode += 1
        data_all[column_temp] = data_all[column_temp].map(dict_temp)
    train_df = data_all.fillna(0)
    return train_df, columns_value_selected, columns_embedd_selected, list(wide_df_dict.keys()), start_encode


def auc_calculate(groundtruth, predicted_prob):
    fpr, tpr, thresholds = roc_curve(groundtruth, predicted_prob, pos_label=1)
    return auc(fpr, tpr)


def train_deep_wide(train_df, columns_value_selected, columns_embedd_selected, wide_selected, bag_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_index = []
    val_index = []

    for i in range(len(train_df)):
        if i % int(1 / opt.valid_portion) == 0:
            val_index.append(i)
        else:
            train_index.append(i)
    val_df = train_df.iloc[val_index]
    train_df = train_df.iloc[train_index]


    train_data_value = torch.tensor(np.array(train_df[columns_value_selected])).float()
    train_data_embedd = torch.tensor(np.array(train_df[columns_embedd_selected])).long()
    train_wide = torch.tensor(np.array(train_df[wide_selected])).float()
    train_data_label = torch.tensor(np.array(train_df['TARGET'])).float()
    torch_dataset = Data.TensorDataset(train_data_value, train_data_embedd, train_wide, train_data_label)
    loader = Data.DataLoader(dataset=torch_dataset, shuffle=True, batch_size=opt.batchSize)

    val_data_value = torch.tensor(np.array(val_df[columns_value_selected])).float()
    val_data_embedd = torch.tensor(np.array(val_df[columns_embedd_selected])).long()
    val_wide = torch.tensor(np.array(val_df[wide_selected])).float()
    val_data_label = torch.tensor(np.array(val_df['TARGET'])).float()
    val_torch_dataset = Data.TensorDataset(val_data_value, val_data_embedd, val_wide, val_data_label)
    val_loader = Data.DataLoader(dataset=val_torch_dataset, shuffle=True, batch_size=opt.batchSize)

    model = CLASS_Wide_Deep(num_embed=len(columns_embedd_selected), num_val=len(columns_value_selected), wide_num=len(wide_selected),
                            bag_size=bag_size).to(device)

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1, last_epoch=-1)
    # criterion_1 = nn.CrossEntropyLoss()
    criterion_1 = torch.nn.BCELoss(size_average=True)

    val_auc_best = 0
    count = 0
    for epoch in range(opt.epoch):
        loss = 0
        for batch_data in loader:
            model.train()
            inputs_value = batch_data[0].to(device)
            inputs_embed = batch_data[1].to(device)
            inputs_wide = batch_data[2].to(device)
            outputs_gt = batch_data[3].to(device)
            optimizer.zero_grad()
            outputs_class = model(inputs_value, inputs_embed, inputs_wide)

            train_class_loss = criterion_1(outputs_class.squeeze(), outputs_gt)
            train_class_loss.backward()
            optimizer.step()
            loss += train_class_loss.item()
        loss = loss/len(loader)
        print('Loss in Epoch {0}: {1}'.format(epoch, loss))

        if epoch % 1 == 0 or epoch == 0:
            count += 1
            auc_val = []
            with torch.no_grad():
                for batch_data in val_loader:
                    model.eval()
                    inputs_value = batch_data[0].to(device)
                    inputs_embed = batch_data[1].to(device)
                    inputs_wide = batch_data[2].to(device)
                    outputs_gt = batch_data[3]
                    optimizer.zero_grad()
                    outputs_class = model(inputs_value, inputs_embed, inputs_wide)
                    # outputs_predit = list(F.softmax(outputs_class, dim=-1).cpu().numpy()[:, 1])
                    outputs_predit = list(outputs_class.squeeze().cpu())
                    auc_val.append(auc_calculate(outputs_gt.numpy(), outputs_predit))
                print('Val AUC in Epoch {0}: {1}'.format(epoch, np.mean(auc_val)))

            if np.mean(auc_val) > val_auc_best:
                val_auc_best = np.mean(auc_val)
                print('Best Val AUC in Epoch {0}: {1}'.format(epoch, val_auc_best))
                best_model = model
                count = 0
            if count > 10:
                print('Early stop')
                break
        schedule.step()
    print('Best Val AUC: {0}'.format(val_auc_best))
    return val_auc_best


def main():
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = data_loaded()
    data_all = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    train_df, columns_value_selected, columns_embedd_selected, wide_selected, bag_size = data_preprocessed(data_all)
    train_deep_wide(train_df, columns_value_selected, columns_embedd_selected, wide_selected, bag_size)


if __name__ == '__main__':
    main()


