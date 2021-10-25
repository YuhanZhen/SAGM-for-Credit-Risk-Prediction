import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.utils.data as Data
import gc
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import warnings
import argparse


warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../../data/',
                    help='the folder to save the raw data')
parser.add_argument('--sample', type=bool, default=True, help='whether to use sample data')
parser.add_argument('--split_portion', type=float, default=0.9, help='the portion of training data')
parser.add_argument('--optimizer', default=optim.Adam, help='optimizer for training')
# parser.add_argument('--optimizer', default=optim.SGD, help='optimizer for training')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--step_size', type=float, default=50, help='the interval epoch to decay learning rate')
parser.add_argument('--EmbeddingSize', type=int, default=5, help='Embedding size for enumerated variable')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--l2', type=float, default=0, help='learning rate decay rate')
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')

opt, unknown = parser.parse_known_args()


class CLASS_Wide_Deep(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_num = opt.EmbeddingSize
        self.embedding_column = nn.Embedding(kwargs['bag_size'], self.embedding_num)
        self.deep_layer_1 = nn.Linear(opt.EmbeddingSize * kwargs['num_embed'] + kwargs['num_val'], 64)
        self.deep_layer_2 = nn.Linear(64, 64)
        self.deep_layer_3 = nn.Linear(64, 32)
        self.deep_layer_4 = nn.Linear(32, 16)
        self.deep_layer_5 = nn.Linear(16, 2)
        self.wide_layer = nn.Linear(kwargs['wide_num'], 2)
        self.output_layer_1 = nn.Linear(2, 1)
        self.output_layer_2 = nn.Linear(2, 1)
        self.dropout_cat = nn.Dropout(0.8)

    def forward(self, value_batch, embedd_batch, wide_batch):
        embedd_batch = self.embedding_column(embedd_batch)
        embedd_layer = embedd_batch.flatten(-2)
        self.concat_deep = torch.cat((embedd_layer, value_batch), 1)

        self.cat_1 = self.dropout_cat(F.relu(self.deep_layer_1(self.concat_deep)))

        self.cat_2 = self.dropout_cat(F.relu(self.deep_layer_2(self.cat_1)))
        self.cat_3 = self.dropout_cat(F.relu(self.deep_layer_3(self.cat_2)))
        self.cat_4 = self.dropout_cat(F.relu(self.deep_layer_4(self.cat_3)))
        self.cat_5 = self.dropout_cat(F.relu(self.deep_layer_5(self.cat_4)))
        self.wide = F.sigmoid(self.wide_layer(wide_batch))

        output_1 = F.sigmoid(self.output_layer_1(torch.cat((self.cat_5[:,0].unsqueeze(1), self.wide[:,0].unsqueeze(1)), 1)))
        output_2 = F.sigmoid(self.output_layer_1(torch.cat((self.cat_5[:, 1].unsqueeze(1), self.wide[:, 1].unsqueeze(1)), 1)))
        self.output = torch.cat((output_1, output_2), 1)
        # self.output = F.sigmoid(self.output_layer(torch.cat((self.cat_5, self.wide), 1)))

        return self.output


def containsAny(seq, aset):
    return True if any(i in seq for i in aset) else False


def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])


def data_preprocessing(accepted_data, ratio_train=0.9):
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
    values_column_list = []
    columns_list = list(accepted_data.columns)
    columns_list.remove('loan_label')
    columns_list.remove('issue_d')

    wide_list = []
    for column_temp in list(columns_list):
        if accepted_data[column_temp].dtype == 'object':
            wide_list.append(column_temp)
        if len(accepted_data[column_temp].unique()) < 200:
            object_column_list.append(column_temp)
        else:
            values_column_list.append(column_temp)

    wide_dict_df = {}
    for column_temp in wide_list:
        temp = pd.get_dummies(accepted_data[column_temp])
        for temp_name in temp.columns.tolist():
            wide_dict_df[column_temp + '_' + str(temp_name)] = list(temp[temp_name])
    accepted_data[list(wide_dict_df.keys())] = wide_dict_df.values()


    accepted_data[values_column_list] = (accepted_data[values_column_list] - accepted_data[values_column_list].min()) / \
                               (accepted_data[values_column_list].max() - accepted_data[values_column_list].min())

    start_encode = 0
    for column_temp in object_column_list:
        dict_temp = {}
        for i in accepted_data[column_temp].unique():
            dict_temp[i] = start_encode
            start_encode += 1
        accepted_data[column_temp] = accepted_data[column_temp].map(dict_temp)

    accepted_data['issue_d'] = pd.to_datetime(accepted_data['issue_d'])
    accepted_data = accepted_data.fillna(0)
    data_train = accepted_data.loc[accepted_data['issue_d'] < accepted_data['issue_d'].quantile(ratio_train)]
    data_test = accepted_data.loc[accepted_data['issue_d'] >= accepted_data['issue_d'].quantile(ratio_train)]

    data_train.drop('issue_d', axis=1, inplace=True)
    data_test.drop('issue_d', axis=1, inplace=True)

    del accepted_data
    gc.collect()
    return data_train, data_test, values_column_list, object_column_list, list(wide_dict_df.keys()), start_encode


def data_load(sample):
    if sample:
        accepted_data = pd.read_csv(os.path.join(opt.path, 'sample/lending_club_sample.csv'), low_memory=False,
                                    parse_dates=['issue_d'], infer_datetime_format=True, encoding='utf_8_sig')
    else:
        accepted_data = pd.read_csv(os.path.join(opt.path, 'lending_club/accepted_2007_to_2018Q4.csv'), low_memory=False,
                                    parse_dates=['issue_d'], infer_datetime_format=True, encoding='utf_8_sig')
    return accepted_data



def auc_calculate(groundtruth, predicted_prob):
    fpr, tpr, thresholds = roc_curve(groundtruth, predicted_prob, pos_label=1)
    return auc(fpr, tpr)


def train_deep_wide(train_data, test_data, columns_value_selected, columns_embedd_selected, wide_selected, bag_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_value = torch.tensor(np.array(train_data[columns_value_selected])).float()
    train_data_embedd = torch.tensor(np.array(train_data[columns_embedd_selected])).long()
    train_wide = torch.tensor(np.array(train_data[wide_selected])).float()
    train_data_label = torch.tensor(np.array(train_data['loan_label'])).long()
    torch_dataset = Data.TensorDataset(train_data_value, train_data_embedd, train_wide, train_data_label)
    loader = Data.DataLoader(dataset=torch_dataset, shuffle=True, batch_size=opt.batchSize)

    val_data_value = torch.tensor(np.array(test_data[columns_value_selected])).float()
    val_data_embedd = torch.tensor(np.array(test_data[columns_embedd_selected])).long()
    val_wide = torch.tensor(np.array(test_data[wide_selected])).float()
    val_data_label = torch.tensor(np.array(test_data['loan_label'])).long()
    val_torch_dataset = Data.TensorDataset(val_data_value, val_data_embedd, val_wide, val_data_label)
    val_loader = Data.DataLoader(dataset=val_torch_dataset, shuffle=True, batch_size=opt.batchSize)

    model = CLASS_Wide_Deep(num_embed=len(columns_embedd_selected), num_val=len(columns_value_selected), wide_num=len(wide_selected),
                            bag_size=bag_size).to(device)

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    # optimizer = opt.optimizer(model.parameters(), lr=opt.lr)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1, last_epoch=-1)
    criterion_1 = nn.CrossEntropyLoss()
    # criterion_1 = torch.nn.BCELoss(size_average=True)
    # criterion_2 = nn.MSELoss()

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
                    outputs_predit = list(F.softmax(outputs_class, dim=-1).cpu().numpy()[:, 1])
                    # outputs_predit = list(outputs_class.squeeze().cpu())
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
    accepted_data = data_load(sample=opt.sample)
    data_train, data_test, columns_value_selected, columns_embedd_selected, wide_selected, bag_size = \
        data_preprocessing(accepted_data, ratio_train=opt.split_portion)
    train_deep_wide(data_train, data_test, columns_value_selected, columns_embedd_selected, wide_selected, bag_size)


if __name__ == '__main__':
    main()


