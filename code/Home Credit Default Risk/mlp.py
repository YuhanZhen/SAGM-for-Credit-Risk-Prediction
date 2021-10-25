#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on October, 2021
@author: ZihaoLi97
"""
import torch
import argparse
import torch.optim as optim
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from data_preprocessing import data_loaded, feature_engineering

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--valid_portion', type=float, default=0.1, help='the portion of validation part in train data')
parser.add_argument('--optimizer', default=optim.Adam, help='optimizer for training')
parser.add_argument('--epoch', type=int, default=200, help='the number of epochs to train for')
parser.add_argument('--step_size', type=float, default=50, help='the interval epoch to decay learning rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0, help='learning rate decay rate')
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')


opt, unknown = parser.parse_known_args()


def containsAny(seq, aset):
    return True if any(i in seq for i in aset) else False


def data_preprocess(df_data):
    list_columns = df_data.columns.tolist()
    list_columns.remove('SK_ID_CURR')
    list_columns.remove('TARGET')
    columns_value_selected = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE',
                              'DAYS_EMPLOYED',
                              'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                              'DAYS_LAST_PHONE_CHANGE', 'index']
    value_column = ['MEAN', 'SUM', 'MIN', 'MAX', 'COUNT', 'RATIO', 'STD', 'AVG', 'TOTAL', 'MODE', 'MEDI']

    columns_embedd_selected = []
    for column_temp in list_columns:
        if containsAny(column_temp, value_column):
            columns_value_selected.append(column_temp)
        elif len(df_data[column_temp].unique()) > 50:
            columns_value_selected.append(column_temp)
        else:
            columns_embedd_selected.append(column_temp)

    df_data[columns_value_selected] = df_data[columns_value_selected].fillna(0)

    df_data[columns_value_selected] = (df_data[columns_value_selected] - df_data[columns_value_selected].min()) / (
            df_data[columns_value_selected].max() - df_data[columns_value_selected].min())
    df_data[columns_value_selected] = df_data[columns_value_selected].fillna(0)
    start_encode = 0

    for column_temp in columns_embedd_selected:
        dict_temp = {}
        for i in df_data[column_temp].unique():
            dict_temp[i] = start_encode
            start_encode += 1
        df_data[column_temp] = df_data[column_temp].map(dict_temp)
    return df_data, columns_value_selected, columns_embedd_selected


def auc_calculate(groundtruth, predicted_prob):
    fpr, tpr, thresholds = roc_curve(groundtruth, predicted_prob, pos_label=1)
    return auc(fpr, tpr)


def error_val_predited(groundtruth, prediction_prob, count):
    predicte_label = []
    for i in prediction_prob:
        if i > 0.5:
            predicte_label.append(1)
        else:
            predicte_label.append(0)
    for i, j in zip(predicte_label, groundtruth):
        if i == 1 and j == 0:
            count['Groundtruth_0-Predicted_1'] += 1
        elif i == 0 and j == 1:
            count['Groundtruth_1-Predicted_0'] += 1
    return count


class CLASS_MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.all_layer_1 = nn.Linear((kwargs['embedd_columns_num'] + kwargs['values_columns_num']), 512)
        self.all_layer_2 = nn.Linear(512, 256)
        self.all_layer_3 = nn.Linear(256, 128)
        self.all_layer_4 = nn.Linear(128, 64)
        self.all_layer_5 = nn.Linear(64, 32)
        self.all_layer_6 = nn.Linear(32, 2)

        self.dropout_layer = nn.Dropout(0.35)

    def forward(self, value_batch, embedd_batch):
        self.all_data = torch.cat((value_batch.float(), embedd_batch.float()), 1)
        self.output_1 = self.dropout_layer(F.relu(self.all_layer_1(self.all_data)))
        self.output_2 = self.dropout_layer(F.relu(self.all_layer_2(self.output_1)))
        self.output_3 = self.dropout_layer(F.relu(self.all_layer_3(self.output_2)))
        self.output_4 = self.dropout_layer(F.relu(self.all_layer_4(self.output_3)))
        self.output_5 = self.dropout_layer(F.relu(self.all_layer_5(self.output_4)))
        self.output = F.sigmoid(self.all_layer_6(self.output_5))

        return self.output


def training_model_mlp(data_all, columns_value, columns_embed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_index = []
    val_index = []
    for i in range(len(data_all)):
        if i % int(1 / opt.valid_portion) == 0:
            val_index.append(i)
        else:
            train_index.append(i)
    val_df = data_all.iloc[val_index]
    train_df = data_all.iloc[train_index]


    train_data_value = torch.tensor(np.array(train_df[columns_value])).float()
    train_data_embedd = torch.tensor(np.array(train_df[columns_embed])).long()
    train_data_label = torch.tensor(np.array(train_df['TARGET'])).long()
    torch_dataset = Data.TensorDataset(train_data_value, train_data_embedd, train_data_label)
    loader = Data.DataLoader(dataset=torch_dataset, shuffle=True, batch_size=opt.batchSize)

    val_data_value = torch.tensor(np.array(val_df[columns_value])).float()
    val_data_embedd = torch.tensor(np.array(val_df[columns_embed])).long()
    val_data_label = torch.tensor(np.array(val_df['TARGET'])).long()
    val_torch_dataset = Data.TensorDataset(val_data_value, val_data_embedd, val_data_label)
    val_loader = Data.DataLoader(dataset=val_torch_dataset, shuffle=True, batch_size=opt.batchSize)

    epochs = opt.epoch
    best_auc_val = 0
    count = 0

    model = CLASS_MLP(values_columns_num=len(columns_value), embedd_columns_num=len(columns_embed)).to(device)
    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1, last_epoch=-1)

    criterion_1 = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = 0
        auc_train = []
        for batch_data in loader:
            model.train()
            inputs_value = batch_data[0].to(device)
            inputs_embed = batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs_value, inputs_embed)
            outputs_class = list(F.softmax(outputs, dim=-1).cpu().detach().numpy()[:, 1])  ## crossentropy loss
            auc_train.append(auc_calculate(batch_data[2].numpy(), outputs_class))
            train_loss = criterion_1(outputs, batch_data[2].to(device))
            train_loss.backward()

            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(loader)

        print('Train loss in Epoch {0}: {1}'.format(epoch, loss))
        print('Train AUC in Epoch {0}: {1}'.format(epoch, np.mean(auc_train)))

        if epoch % 1 == 0 or epoch == 0:
            count += 1
            auc_val = []
            dict_error = {'Groundtruth_0-Predicted_1': 0, 'Groundtruth_1-Predicted_0': 0}
            with torch.no_grad():
                val_loss = 0
                for batch_data in val_loader:
                    model.eval()
                    inputs_value = batch_data[0].to(device)
                    inputs_embed = batch_data[1].to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs_value, inputs_embed)
                    output_class = list(F.softmax(outputs, dim=-1).cpu().numpy()[:, 1])  ## crossentropy loss
                    val_loss = criterion_1(outputs, batch_data[2].to(device))
                    val_loss += val_loss.item()
                    auc_val.append(auc_calculate(batch_data[2].numpy(), output_class))
                    dict_error = error_val_predited(batch_data[2].numpy(), output_class, dict_error)
                val_loss = val_loss / len(val_loader)
                print('Val loss in Epoch {0}: {1}'.format(epoch, val_loss))

                print('Val AUC in Epoch {0}: {1}'.format(epoch, np.mean(auc_val)))
                print(dict_error)

            if np.mean(auc_val) > best_auc_val:
                best_auc_val = np.mean(auc_val)
                print('Best AUC in Epoch {0}: {1}'.format(epoch, best_auc_val))
                best_model = model
                count = 0
            if count > 10:
                print('Save epoch {0}'.format(epoch))
                print('Early stop')
                break
        schedule.step()


def main():
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = data_loaded()
    data_all = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    data_all, columns_value, columns_embed = data_preprocess(data_all)
    val_dict = training_model_mlp(data_all, columns_value, columns_embed)
    print(val_dict)


if __name__ == '__main__':
    main()
