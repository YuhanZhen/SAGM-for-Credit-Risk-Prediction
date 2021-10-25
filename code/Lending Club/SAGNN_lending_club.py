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
import time
import random
import gc

import pickle

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../../data/', help='the folder to save the training and test raw data')
parser.add_argument('--sample', type=bool, default=True, help='whether to use sample data')
parser.add_argument('--cluster_theta', type=int, default=10, help='the parameter to control graph construction')
parser.add_argument('--valid_portion', type=float, default=0.9, help='the portion of training data')
parser.add_argument('--optimizer', default=optim.Adam, help='optimizer for training')
parser.add_argument('--epoch', type=int, default=200, help='the number of epochs to train for')
parser.add_argument('--step_size', type=float, default=50, help='the interval epoch to decay learning rate')
parser.add_argument('--EmbeddingSize', type=int, default=5, help='Embedding size for enumerated variable')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0, help='learning rate decay rate')
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
parser.add_argument('--lambda_', type=float, default=0.5,
                    help='the parameter for multitasks learning (classification)')
parser.add_argument('--alpha_', type=float, default=0.5,
                    help='the parameter for multitasks learning (input reconstruction)')
parser.add_argument('--beta_', type=float, default=0.0, help='the parameter for multitasks learning (cosine similarity between positive and negative intermediate features)')
parser.add_argument('--early_stop_epoch', type=int, default=10, help='the parameter to control the early stop epochs')
parser.add_argument('--intermediate_vector_save', type=bool, default=False, help='whether to save the intermediate vector')

opt, unknown = parser.parse_known_args()

f = open(os.path.join(opt.path, 'model_param_training.txt'), 'w')
f.writelines('-----------------parameter----------------------' + '\n')
argsDict = opt.__dict__
for arg_temp, value in argsDict.items():
    f.writelines(arg_temp + ': ' + str(value) + '\n')


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


def data_preprocessing(accepted_data):
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

    for column_temp in list(columns_list):
        if len(accepted_data[column_temp].unique()) < 200:
            object_column_list.append(column_temp)
        else:
            values_column_list.append(column_temp)

    accepted_data[values_column_list] = (accepted_data[values_column_list] - accepted_data[values_column_list].min()) / \
                               (accepted_data[values_column_list].max() - accepted_data[values_column_list].min())

    start_encode = 0
    for column_temp in object_column_list:
        dict_temp = {}
        for i in accepted_data[column_temp].unique():
            dict_temp[i] = start_encode
            start_encode += 1
        accepted_data[column_temp] = accepted_data[column_temp].map(dict_temp)

    return accepted_data, values_column_list, object_column_list, start_encode


def data_cluster(data_all, columns_object, ratio_train=0.9):
    data_all['id'] = [i for i in range(len(data_all))]
    data_all[columns_object] = data_all[columns_object].fillna(0)
    dict_group_ioc = {}
    groups = data_all.groupby(columns_object)
    count = 0
    for temp_group in groups:
        temp = list(temp_group[1]['id'])
        dict_group_ioc_copy = dict_group_ioc
        dict_group_ioc_copy.update(dict(zip(temp, [count] * len(temp))))
        count += 1
    clusters = data_all['id'].map(dict_group_ioc_copy).tolist()
    data_all['clusters_group'] = clusters
    data_all['issue_d'] = pd.to_datetime(data_all['issue_d'])
    accepted_data = data_all.fillna(0)
    data_train = accepted_data.loc[accepted_data['issue_d'] < accepted_data['issue_d'].quantile(ratio_train)]
    data_test = accepted_data.loc[accepted_data['issue_d'] >= accepted_data['issue_d'].quantile(ratio_train)]
    data_train.drop('issue_d', axis=1, inplace=True)
    data_test.drop('issue_d', axis=1, inplace=True)
    del accepted_data
    gc.collect()

    train_cluster = list(data_train['clusters_group'])
    test_cluster = list(data_test['clusters_group'])
    data_train = data_train.drop(['id', 'clusters_group'], axis=1)
    data_test = data_test.drop(['id', 'clusters_group'], axis=1)
    return train_cluster, test_cluster, data_train, data_test


def intermediate_feature_distance(intermediate_features, label_batch):
    positive_vector = torch.mean(intermediate_features * label_batch.unsqueeze(-1).float(), dim=0)
    zero = torch.zeros_like(label_batch)
    label_temp = label_batch + 1
    label_negative = torch.where(label_temp==2, zero, label_temp)
    negative_vector = torch.mean(intermediate_features * label_negative.unsqueeze(-1).float(), dim=0)
    similarity = abs(torch.cosine_similarity(positive_vector, negative_vector, dim=0))
    return similarity


def matrix_connection(a, device='cuda'):
    a = a.to('cpu')
    a_array = a.numpy()
    dict_index = {}
    for i in a.unique():
        dict_index[i.numpy().tolist()] = sum(np.argwhere(a_array == i.numpy()).tolist(), [])
    matrix_connect = np.zeros((len(a), len(a)))
    degree_matrix = np.zeros((len(a), len(a)))
    for index_column, i in enumerate(a):
        for j in dict_index[i.numpy().tolist()]:
            matrix_connect[index_column][j] = 1
        degree_matrix[index_column][index_column] = len(dict_index[i.numpy().tolist()])
    matrix_connect = torch.tensor(matrix_connect)
    degree_matrix = torch.inverse(torch.sqrt(torch.tensor(degree_matrix)))
    return matrix_connect.to(device), degree_matrix.to(device)


class CLASS_NN_Embed_cluster(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_num = opt.EmbeddingSize
        self.embedding_column = nn.Embedding(kwargs['bag_size'], self.embedding_num)
        self.q = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.k = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.v = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.att_dropout = nn.Dropout(0.35)
        self.layer_norm_att = nn.LayerNorm(self.embedding_num * kwargs['embedd_columns_num'], eps=1e-6)

        self.q_value = nn.Linear(kwargs['values_columns_num'], 16, bias=False)
        self.k_value = nn.Linear(kwargs['values_columns_num'], 16, bias=False)
        self.v_value = nn.Linear(kwargs['values_columns_num'], 16, bias=False)
        self.value_layer_norm_att = nn.LayerNorm(16, eps=1e-6)

        self.layer_concat_1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 16, 32)
        self.layer_concat_2 = nn.Linear(32 + kwargs['values_columns_num'], 32)
        self.layer_concat_3 = nn.Linear(32, 16)
        self.gnn_concat = nn.Linear(16, 16)
        self.layer_concat_4 = nn.Linear(16, 2)

        self.decoder_1 = nn.Linear(kwargs['embedd_columns_num'] * self.embedding_num + 16, 32)
        self.decoder_2 = nn.Linear(32, kwargs['embedd_columns_num'] + kwargs['values_columns_num'])

        self.gnn_layer_1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 16,
                                     self.embedding_num * kwargs['embedd_columns_num'] + 16)
        self.gnn_layer_2 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 16,
                                     self.embedding_num * kwargs['embedd_columns_num'] + 16)

        self.alpha_attention = torch.nn.Parameter(torch.randn(1))
        self.layer_dropout = nn.Dropout(0.35)

    def forward(self, value_batch, embedd_batch, clusters):
        connections_matrix, degree_matrix = matrix_connection(clusters)
        embedd_batch = self.embedding_column(embedd_batch)
        query_layer = self.q(embedd_batch)
        key_layer = self.k(embedd_batch)
        value_layer = self.v(embedd_batch)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.att_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.layer_norm_att(context_layer.flatten(-2))

        value_query_layer = self.q_value(value_batch)
        value_key_layer = self.k_value(value_batch)
        value_value_layer = self.v_value(value_batch)
        value_attention_scores = nn.Softmax(dim=-1)(value_query_layer * value_key_layer)
        value_attention_probs = self.att_dropout(value_attention_scores)
        value_context_layer = (value_attention_probs * value_value_layer)
        value_context_layer = self.value_layer_norm_att(value_context_layer)

        self.output = torch.cat((context_layer, value_context_layer), 1)

        connection = torch.matmul(torch.matmul(degree_matrix.float(), connections_matrix.float()), degree_matrix.float())
        self.gnn_1 = F.relu(self.gnn_layer_1(torch.matmul(connection.float(), self.output.float())))
        self.gnn = F.relu(self.gnn_layer_2(torch.matmul(connection.float(), self.gnn_1)))

        self.output_0 = F.relu(self.layer_concat_1(self.alpha_attention * self.output + (1 - self.alpha_attention) * self.gnn))
        self.output_0_df = self.layer_dropout(self.output_0)
        self.output_1 = torch.cat((self.output_0_df, value_batch), 1)
        self.output_2 = self.layer_dropout(F.relu(self.layer_concat_2(self.output_1)))
        self.output_3 = self.layer_dropout(F.relu(self.layer_concat_3(self.output_2)))
        self.output_4 = F.relu(self.layer_concat_4(self.output_3))

        self.decoder_val_1 = self.layer_dropout(F.relu(self.decoder_1(self.output)))
        self.decoder_val_2 = F.relu(self.decoder_2(self.decoder_val_1))

        return self.output_4, self.decoder_val_2, self.output_2


def auc_calculate(groundtruth, predicted_prob):
    fpr, tpr, thresholds = roc_curve(groundtruth, predicted_prob, pos_label=1)
    return auc(fpr, tpr)


def training_model_classification(train_df, val_df, train_cluster, val_cluster, value_column, embed_column, bag_size, f):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





    train_data_value = torch.tensor(np.array(train_df[value_column])).float()
    train_data_embedd = torch.tensor(np.array(train_df[embed_column])).long()
    train_clusters = torch.tensor(np.array(train_cluster))
    train_data_label = torch.tensor(np.array(train_df['loan_label'])).long()
    torch_dataset = Data.TensorDataset(train_data_value, train_data_embedd, train_clusters, train_data_label)
    loader = Data.DataLoader(dataset=torch_dataset, shuffle=True, batch_size=opt.batchSize)


    val_data_value = torch.tensor(np.array(val_df[value_column])).float()
    val_data_embedd = torch.tensor(np.array(val_df[embed_column])).long()
    val_clusters = torch.tensor(np.array(val_cluster))
    val_data_label = torch.tensor(np.array(val_df['loan_label'])).long()
    val_torch_dataset = Data.TensorDataset(val_data_value, val_data_embedd, val_clusters, val_data_label)
    val_loader = Data.DataLoader(dataset=val_torch_dataset, shuffle=True, batch_size=opt.batchSize)

    epochs = opt.epoch
    model = CLASS_NN_Embed_cluster(embedd_columns_num=len(embed_column), values_columns_num=len(value_column),
                                   bag_size=bag_size).to(device)

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1, last_epoch=-1)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()
    lam = opt.lambda_
    alp = opt.alpha_
    beta = opt.beta_

    best_auc_val = 0
    count = 0
    flag_earlystop = 0
    for epoch in range(epochs):
        loss = 0
        for batch_data in loader:
            model.train()
            inputs_value = batch_data[0].to(device)
            inputs_embed = batch_data[1].to(device)
            inputs_cluster = batch_data[2].to(device)
            optimizer.zero_grad()
            outputs_class, outputs_ae, intermediate_vector = model(inputs_value, inputs_embed, inputs_cluster)
            train_loss_class = criterion_1(outputs_class, batch_data[3].to(device))
            train_loss_ae = criterion_2(outputs_ae, torch.cat((batch_data[0].float(), batch_data[1].float()), -1).to(device))
            train_loss_ae /= batch_data[0].size()[0]
            # regularition = torch.norm(input=torch.matmul(model.context_layer, model.context_layer.transpose(-1, -2)) - torch.eye(model.context_layer.shape[1]).to(device), p='fro')
            cosine_sim = intermediate_feature_distance(intermediate_vector, batch_data[3].to(device))
            train_loss = lam * train_loss_class + alp * train_loss_ae + beta * cosine_sim

            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(loader)
        f.writelines('Loss in Epoch {0}: {1}'.format(epoch, loss) + '\n')
        print('Loss in Epoch {0}: {1}'.format(epoch, loss))

        count += 1
        auc_val = []
        with torch.no_grad():
            for batch_data in val_loader:
                model.eval()
                inputs_value = batch_data[0].to(device)
                inputs_embed = batch_data[1].to(device)
                inputs_cluster = batch_data[2].to(device)
                optimizer.zero_grad()
                outputs, _, _ = model(inputs_value, inputs_embed, inputs_cluster)
                output_train = list(F.softmax(outputs, dim=-1).cpu().numpy()[:, 1])  ## crossentropy loss
                auc_val.append(auc_calculate(batch_data[3].numpy(), output_train))
            print('Val AUC in Epoch {0}: {1}'.format(epoch, np.mean(auc_val)))
            f.writelines('Val AUC in Epoch {0}: {1}'.format(epoch, np.mean(auc_val)) + '\n')

        if np.mean(auc_val) > best_auc_val:
            best_auc_val = np.mean(auc_val)
            print('Best Val AUC in Epoch {0}: {1}'.format(epoch, best_auc_val))
            f.writelines('Best Val AUC in Epoch {0}: {1}'.format(epoch, best_auc_val) + '\n')
            best_model = model
            count = 0

        if count > opt.early_stop_epoch:
            flag_earlystop = 1
            print('Save epoch {0}'.format(epoch))
            f.writelines('Save epoch {0}'.format(epoch) + '\n')
            path_model = os.path.join(opt.path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_early_stop_best_model_' + str(epoch) + '.pth')
            torch.save(best_model.state_dict(), path_model)
            f.writelines('Early stop' + '\n')
            print('Early stop')
            break
        schedule.step()

    if flag_earlystop == 0:
        path_model = os.path.join(opt.path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_best_model_' + str(epoch) + '.pth')
        torch.save(best_model.state_dict(), path_model)

    if opt.intermediate_vector_save:
        train_loader_copy = Data.DataLoader(dataset=torch_dataset, shuffle=False, batch_size=opt.batchSize)
        val_loader_copy = Data.DataLoader(dataset=val_torch_dataset, shuffle=False, batch_size=opt.batchSize)
        train_feature = {'feature_intermediate': []}
        val_feature = {'feature_intermediate': []}
        with torch.no_grad():
            for batch_data in train_loader_copy:
                best_model.eval()
                inputs_value = batch_data[0].to(device)
                inputs_embed = batch_data[1].to(device)
                inputs_cluster = batch_data[2].to(device)
                optimizer.zero_grad()
                outputs, _, _ = best_model(inputs_value, inputs_embed, inputs_cluster)
                train_feature['feature_intermediate'] += best_model.output_0.cpu().numpy().tolist()

            for batch_data in val_loader_copy:
                best_model.eval()
                inputs_value = batch_data[0].to(device)
                inputs_embed = batch_data[1].to(device)
                inputs_cluster = batch_data[2].to(device)
                optimizer.zero_grad()
                outputs, _, _ = best_model(inputs_value, inputs_embed, inputs_cluster)
                val_feature['feature_intermediate'] += best_model.output_0.cpu().numpy().tolist()

            with open(os.path.join(opt.path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + 'train_features_intermediate.pkl'), 'wb') as f:
                pickle.dump(train_feature, f, pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(opt.path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + 'val_features_intermediate.pkl'), 'wb') as f:
                pickle.dump(val_feature, f, pickle.HIGHEST_PROTOCOL)


    return best_auc_val


def cluster_analysis_uniq(data_all, cluster_attributes):
    dict_column = {}
    for column in list(data_all.columns):
        dict_column[column] = len(list(data_all[column].unique()))
    selected_attributes = []
    for column_temp in dict_column.keys():
        if dict_column[column_temp] < cluster_attributes:
            selected_attributes.append(column_temp)
    return selected_attributes


def main():
    accepted_data = data_load(sample=opt.sample)
    data_all, columns_value, columns_embed, bag_size = data_preprocessing(accepted_data)
    selected_attributes = cluster_analysis_uniq(data_all, opt.cluster_theta)
    train_cluster, test_cluster, train_df, test_df = data_cluster(data_all, selected_attributes, ratio_train=opt.valid_portion)
    val_best = training_model_classification(train_df, test_df, train_cluster, test_cluster, columns_value,
                                             columns_embed, bag_size, f)

    print('Best AUC: {0}'.format(np.round(val_best, 5)))
    f.close()


if __name__ == '__main__':
    main()



