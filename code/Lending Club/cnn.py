import torch
import os
import pandas as pd
import argparse
import torch.optim as optim
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import gc
import random

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../../data/', help='the folder to save the raw data')
parser.add_argument('--sample', type=bool, default=True, help='whether to use sample data')
parser.add_argument('--split_portion', type=float, default=0.9, help='the portion of training data')
parser.add_argument('--optimizer', default=optim.Adam, help='optimizer for training')
parser.add_argument('--epoch', type=int, default=200, help='the number of epochs to train for')
parser.add_argument('--step_size', type=float, default=50, help='the interval epoch to decay learning rate')
parser.add_argument('--EmbeddingSize', type=int, default=5, help='Embedding size for enumerated variable')
parser.add_argument('--kernel_num', type=int, default=10, help='The number of kernel')
parser.add_argument('--kernel_sizes', type=list, default=[3, 5, 7, 9, 15, 21], help='Kernel size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0, help='learning rate decay rate')
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')


opt, unknown = parser.parse_known_args()


def containsAny(seq, aset):
    return True if any(i in seq for i in aset) else False


def data_load(sample):
    if sample:
        accepted_data = pd.read_csv(os.path.join(opt.path, 'sample/lending_club_sample.csv'), low_memory=False,
                                    parse_dates=['issue_d'], infer_datetime_format=True, encoding='utf_8_sig')
    else:
        accepted_data = pd.read_csv(os.path.join(opt.path, 'lending_club/accepted_2007_to_2018Q4.csv'), low_memory=False,
                                    parse_dates=['issue_d'], infer_datetime_format=True, encoding='utf_8_sig')
    return accepted_data


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

    for column_temp in list(accepted_data.columns):
        if accepted_data[column_temp].dtype == 'object':
            accepted_data[column_temp] = accepted_data[column_temp].astype('category').cat.codes

    columns_list = list(accepted_data.columns)
    columns_list.remove('loan_label')
    columns_list.remove('issue_d')
    accepted_data[columns_list] = (accepted_data[columns_list] - accepted_data[columns_list].min())\
                                  / (accepted_data[columns_list].max() - accepted_data[columns_list].min())

    accepted_data['issue_d'] = pd.to_datetime(accepted_data['issue_d'])
    accepted_data = accepted_data.fillna(0)

    data_train = accepted_data.loc[accepted_data['issue_d'] < accepted_data['issue_d'].quantile(ratio_train)]
    data_test = accepted_data.loc[accepted_data['issue_d'] >= accepted_data['issue_d'].quantile(ratio_train)]

    data_train.drop('issue_d', axis=1, inplace=True)
    data_test.drop('issue_d', axis=1, inplace=True)

    del accepted_data
    gc.collect()
    return data_train, data_test


class CLASS_CNN(nn.Module):
    def __init__(self, size_data):
        super().__init__()
        Ci = 1
        out_channels = opt.kernel_num
        Ks = opt.kernel_sizes
        self.convs = nn.ModuleList([nn.Conv1d(Ci, out_channels, K, padding=int(K/2)) for K in Ks])
        # self.concat_layer_1 = nn.Linear(opt.kernel_num * len(opt.kernel_sizes), 32)
        self.concat_layer_1 = nn.Linear(size_data, 64)
        self.concat_layer_2 = nn.Linear(64, 32)
        self.concat_layer_3 = nn.Linear(32, 16)
        self.concat_layer_4 = nn.Linear(16, 8)
        self.concat_layer_5 = nn.Linear(8, 2)
        self.layer_dropout = nn.Dropout(0.35)
        self.dropout = nn.Dropout(0.35)

    def forward(self, value_batch):
        x = [F.relu(conv(value_batch.unsqueeze(1))).transpose(1, 2) for conv in self.convs]
        x = [F.avg_pool1d(temp, temp.size(2)).transpose(1, 2).squeeze(1) for temp in x]
        init_x = x[0].unsqueeze(1)
        for line in x[1:]:
            init_x = torch.cat((init_x, line.unsqueeze(1)), dim=1)
        init_x = init_x.transpose(1, 2)
        x = F.avg_pool1d(init_x, init_x.size(2)).transpose(1, 2).squeeze(1)

        # x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        # x = torch.cat(x, 1)

        x = self.dropout(x)

        output = self.layer_dropout(F.relu(self.concat_layer_1(x)))
        output = self.layer_dropout(F.relu(self.concat_layer_2(output)))
        output = self.layer_dropout(F.relu(self.concat_layer_3(output)))
        output = self.layer_dropout(F.relu(self.concat_layer_4(output)))
        self.output = F.relu(self.concat_layer_5(output))
        return self.output


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


def training_model_cnn(train_data, test_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y_train = train_data['loan_label']
    X_train = train_data.drop(['loan_label'], axis=1)
    y_test = test_data['loan_label']
    X_test = test_data.drop(['loan_label'], axis=1)
    del train_data
    gc.collect()

    train_data_value = torch.tensor(np.array(X_train)).float()
    train_data_label = torch.tensor(np.array(y_train)).long()
    torch_dataset = Data.TensorDataset(train_data_value, train_data_label)
    loader = Data.DataLoader(dataset=torch_dataset, shuffle=True, batch_size=opt.batchSize)
    val_data_value = torch.tensor(np.array(X_test)).float()
    val_data_label = torch.tensor(np.array(y_test)).long()
    val_torch_dataset = Data.TensorDataset(val_data_value, val_data_label)
    val_loader = Data.DataLoader(dataset=val_torch_dataset, shuffle=True, batch_size=opt.batchSize)

    epochs = opt.epoch
    best_auc_val = 0
    count = 0

    model = CLASS_CNN(train_data_value.size(1)).to(device)
    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1, last_epoch=-1)

    criterion_1 = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = 0
        auc_train = []
        for batch_data in loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(batch_data[0].to(device))
            outputs_class = list(F.softmax(outputs, dim=-1).cpu().detach().numpy()[:, 1])  ## crossentropy loss
            auc_train.append(auc_calculate(batch_data[1].numpy(), outputs_class))
            train_loss = criterion_1(outputs, batch_data[1].to(device))
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
                    optimizer.zero_grad()
                    outputs = model(batch_data[0].to(device))
                    output_class = list(F.softmax(outputs, dim=-1).cpu().numpy()[:, 1])  ## crossentropy loss
                    val_loss = criterion_1(outputs, batch_data[1].to(device))
                    val_loss += val_loss.item()
                    auc_val.append(auc_calculate(batch_data[1].numpy(), output_class))
                    dict_error = error_val_predited(batch_data[1].numpy(), output_class, dict_error)
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
    accepted_data = data_load(sample=False)
    data_train, data_test = data_preprocessing(accepted_data, ratio_train=opt.split_portion)
    val_dict = training_model_cnn(data_train, data_test)
    print(val_dict)


if __name__ == '__main__':
    main()

