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
from data_preprocessing import data_loaded, feature_engineering
import copy
import pickle

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data_load_path', default='../../data/home-credit-default-risk-raw/',
                    help='the folder to save the training and test raw data')

parser.add_argument('--theta_m', type=float, default=0.3, help='the ratio of missing value data to all data')
parser.add_argument('--theta_u', type=int, default=20, help='the number of distinct values for the attribute')
parser.add_argument('--up_sample', type=float, default=0.0, help='the ratio for the number of positive samples after and before upsampling')
parser.add_argument('--down_sample', type=float, default=0.0, help='the ratio for the number of negative samples after and before downsampling')
parser.add_argument('--valid_portion', type=float, default=0.1, help='the portion of validation part in train data')
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
parser.add_argument('--beta_', type=float, default=0.0,
                    help='the parameter for multitasks learning (cosine similarity between positive and negative intermediate features)')

opt, unknown = parser.parse_known_args()

f = open(os.path.join(opt.data_load_path, 'model_param_training.txt'), 'w')
f.writelines('-----------------parameter----------------------' + '\n')
argsDict = opt.__dict__
for arg_temp, value in argsDict.items():
    f.writelines(arg_temp + ': ' + str(value) + '\n')


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
    return df_data, columns_value_selected, columns_embedd_selected, start_encode


def data_cluster(df_data, columns_object):
    df_data[columns_object] = df_data[columns_object].fillna(0)
    dict_group_ioc = {}
    groups = df_data.groupby(columns_object)
    count = 0
    for temp_group in groups:
        temp = list(temp_group[1]['SK_ID_CURR'])
        dict_group_ioc_copy = dict_group_ioc
        dict_group_ioc_copy.update(dict(zip(temp, [count] * len(temp))))
        count += 1
    clusters = df_data['SK_ID_CURR'].map(dict_group_ioc_copy).tolist()
    return clusters


def SMOTE_data(train_df):
    label = train_df['TARGET']
    columns = list(train_df.columns)
    columns_copy = copy.copy(columns)
    columns.remove('TARGET')
    data = train_df[columns]
    sm = SMOTE(sampling_strategy=1, random_state=42)
    X_res, y_res = sm.fit_resample(data, label)
    train_df = pd.DataFrame(data=X_res, columns=columns)
    train_df['TARGET'] = y_res
    train_df = train_df.reindex(columns=columns_copy)
    return train_df


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
        self.q = nn.Linear(self.embedding_num, self.embedding_num)
        self.k = nn.Linear(self.embedding_num, self.embedding_num)
        self.v = nn.Linear(self.embedding_num, self.embedding_num)
        self.att_dropout = nn.Dropout(0.35)
        self.layer_norm_att = nn.LayerNorm(self.embedding_num * kwargs['embedd_columns_num'], eps=1e-6)

        self.flatten_nn = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'], 512)
        self.dropout_att = nn.Dropout(0.35)

        self.q_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.k_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.v_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.value_layer_norm_att = nn.LayerNorm(256, eps=1e-6)


        self.layer_concat_1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256, 512)
        self.layer_concat_2 = nn.Linear(512 + kwargs['values_columns_num'], 128)
        self.layer_concat_3 = nn.Linear(128, 32)
        self.gnn_concat = nn.Linear(64, 32)
        self.layer_concat_4 = nn.Linear(32, 2)

        self.decoder_1 = nn.Linear(kwargs['embedd_columns_num'] * self.embedding_num + 256, 1024)
        self.decoder_2 = nn.Linear(1024, kwargs['embedd_columns_num'] + kwargs['values_columns_num'])

        self.gnn_layer_1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256,
                                     self.embedding_num * kwargs['embedd_columns_num'] + 256)
        self.gnn_layer_2 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256,
                                     self.embedding_num * kwargs['embedd_columns_num'] + 256)

        # self.gnn_layer_1 = nn.Linear(32, 32)
        # self.gnn_layer_2 = nn.Linear(32, 32)
        self.alpha_attention = torch.nn.Parameter(torch.randn(1))
        # # self.layer_concat_5 = nn.Linear(32, 1) ## Focal loss
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
        # embedd_layer = self.dropout_att(F.relu(self.flatten_nn(context_layer)))

        value_query_layer = self.q_value(value_batch)
        value_key_layer = self.k_value(value_batch)
        value_value_layer = self.v_value(value_batch)
        value_attention_scores = nn.Softmax(dim=-1)(value_query_layer * value_key_layer)
        value_attention_probs = self.att_dropout(value_attention_scores)
        value_context_layer = (value_attention_probs * value_value_layer)
        value_context_layer = self.value_layer_norm_att(value_context_layer)

        self.output = torch.cat((context_layer, value_context_layer), 1)

        connection = torch.matmul(torch.matmul(degree_matrix.float(), connections_matrix.float()),
                                  degree_matrix.float())
        self.gnn_1 = F.relu(self.gnn_layer_1(torch.matmul(connection.float(), self.output.float())))
        self.gnn = F.relu(self.gnn_layer_2(torch.matmul(connection.float(), self.gnn_1)))

        # gnn_attention_scores = nn.Softmax(dim=-1)(self.gnn * self.output)
        # gnn_attention_probs = self.att_dropout(gnn_attention_scores)
        # gnn_context_layer = (gnn_attention_probs * self.output)
        # self.output_1 = self.layer_dropout(F.relu(self.layer_concat_1(gnn_context_layer)))
        self.output_0 = F.relu(self.layer_concat_1(self.alpha_attention * self.output + (1 - self.alpha_attention) * self.gnn))
        self.output_0_df = self.layer_dropout(self.output_0)

        self.output_1 = torch.cat((self.output_0_df, value_batch), 1)
        self.output_2 = self.layer_dropout(F.relu(self.layer_concat_2(self.output_1)))
        self.output_3 = self.layer_dropout(F.relu(self.layer_concat_3(self.output_2)))


        # self.gnn_1 = F.relu(self.gnn_layer_1(torch.matmul(connection.float(), self.output_3.float())))
        # self.output_3_gnn = F.relu(self.gnn_layer_2(torch.matmul(connection.float(), self.gnn_1)))

        self.output_4 = F.relu(self.layer_concat_4(self.output_3))

        self.decoder_val_1 = self.layer_dropout(F.relu(self.decoder_1(self.output)))
        self.decoder_val_2 = F.relu(self.decoder_2(self.decoder_val_1))

        return self.output_4, self.decoder_val_2, self.output_2


def auc_calculate(groundtruth, predicted_prob):
    fpr, tpr, thresholds = roc_curve(groundtruth, predicted_prob, pos_label=1)
    return auc(fpr, tpr)


def training_model_classification(data_all, clusters, value_column, embed_column, bag_size, f):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.up_sample:
        positive_index = []
        negative_index = []
        for i in range(len(data_all)):
            if data_all.iloc[i]['TARGET'] == 1:
                positive_index.append(i)
            else:
                negative_index.append(i)

        positive_df = data_all.iloc[positive_index]
        negative_df = data_all.iloc[negative_index]
        positive_df = pd.DataFrame(np.repeat(positive_df.values, opt.up_sample, axis=0))
        data_all = pd.concat([positive_df, negative_df], axis=0)
        clusters = [clusters[i] for i in
                          [index_temp for index_temp in positive_index for i in range(opt.up_sample)] + negative_index]
    if opt.down_sample:
        positive_index = []
        negative_index = []
        for i in range(len(data_all)):
            if data_all.iloc[i]['TARGET'] == 1:
                positive_index.append(i)
            else:
                negative_index.append(i)
        positive_df = data_all.iloc[positive_index]

        sample_index_negative = random.sample(negative_index, int(opt.down_sample * len(negative_index)))
        negative_df = data_all.iloc[sample_index_negative]

        data_all = pd.concat([positive_df, negative_df], axis=0)
        clusters = [clusters[i] for i in positive_index+sample_index_negative]

    train_index = []
    val_index = []
    train_clusters = []
    val_clusters = []
    for i, cluster_temp in enumerate(clusters):
        if i % int(1 / opt.valid_portion) == 0:
            val_index.append(i)
            val_clusters.append(cluster_temp)
        else:
            train_index.append(i)
            train_clusters.append(cluster_temp)

    val_df = data_all.iloc[val_index]
    train_df = data_all.iloc[train_index]

    train_data_value = torch.tensor(np.array(train_df[value_column])).float()
    train_data_embedd = torch.tensor(np.array(train_df[embed_column])).long()
    train_clusters = torch.tensor(np.array(train_clusters))
    train_data_label = torch.tensor(np.array(train_df['TARGET'])).long()
    torch_dataset = Data.TensorDataset(train_data_value, train_data_embedd, train_clusters, train_data_label)
    loader = Data.DataLoader(dataset=torch_dataset, shuffle=True, batch_size=opt.batchSize)
    train_loader_copy = Data.DataLoader(dataset=torch_dataset, shuffle=False, batch_size=opt.batchSize)

    val_data_value = torch.tensor(np.array(val_df[value_column])).float()
    val_data_embedd = torch.tensor(np.array(val_df[embed_column])).long()
    val_clusters = torch.tensor(np.array(val_clusters))
    val_data_label = torch.tensor(np.array(val_df['TARGET'])).long()
    val_torch_dataset = Data.TensorDataset(val_data_value, val_data_embedd, val_clusters, val_data_label)
    val_loader = Data.DataLoader(dataset=val_torch_dataset, shuffle=True, batch_size=opt.batchSize)
    val_loader_copy = Data.DataLoader(dataset=val_torch_dataset, shuffle=False, batch_size=opt.batchSize)

    epochs = opt.epoch
    lam = opt.lambda_
    alp = opt.alpha_
    beta = opt.beta_

    model = CLASS_NN_Embed_cluster(embedd_columns_num=len(embed_column), values_columns_num=len(value_column),
                                   bag_size=bag_size).to(device)

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1, last_epoch=-1)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    # criterion = FocalLoss()
    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()

    flag_early_stop = False
    best_auc_val = 0
    count = 0
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
            cosine_sim = intermediate_feature_distance(intermediate_vector, batch_data[2].to(device))
            train_loss = lam * train_loss_class + alp * train_loss_ae + beta * cosine_sim

            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(loader)
        f.writelines('Loss in Epoch {0}: {1}'.format(epoch, loss) + '\n')
        print('Loss in Epoch {0}: {1}'.format(epoch, loss))
        schedule.step()

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
                # output_train = list(outputs.cpu().numpy()) ## Focal loss
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

        if count > 10:
            flag_early_stop = True
            print('Save epoch {0}'.format(epoch))
            f.writelines('Save epoch {0}'.format(epoch) + '\n')
            path_model = os.path.join(opt.data_load_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_early_stop_best_model_' + str(epoch) + '.pth')
            torch.save(best_model.state_dict(), path_model)
            f.writelines('Early stop' + '\n')
            print('Early stop')
            break

    if not flag_early_stop:
        path_model = os.path.join(opt.data_load_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_best_model_' + str(epoch) + '.pth')
        torch.save(best_model.state_dict(), path_model)


    train_feature = {'feature_intermediate': []}
    val_feature = {'feature_intermediate': []}
    with torch.no_grad():
        for batch_data in train_loader_copy:
            best_model.eval()
            inputs_value = batch_data[0].to(device)
            inputs_embed = batch_data[1].to(device)
            inputs_cluster = batch_data[3].to(device)
            optimizer.zero_grad()
            outputs, _, _ = best_model(inputs_value, inputs_embed, inputs_cluster)
            train_feature['feature_intermediate'] += best_model.output_0.cpu().numpy().tolist()

        for batch_data in val_loader_copy:
            best_model.eval()
            inputs_value = batch_data[0].to(device)
            inputs_embed = batch_data[1].to(device)
            inputs_cluster = batch_data[3].to(device)
            optimizer.zero_grad()
            outputs, _, _ = best_model(inputs_value, inputs_embed, inputs_cluster)
            val_feature['feature_intermediate'] += best_model.output_0.cpu().numpy().tolist()

        with open(os.path.join(opt.data_load_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + 'train_features_intermediate.pkl'), 'wb') as f:
            pickle.dump(train_feature, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(opt.data_load_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + 'val_features_intermediate.pkl'), 'wb') as f:
            pickle.dump(val_feature, f, pickle.HIGHEST_PROTOCOL)

    return best_auc_val


def cluster_analysis(data_all, theta_m, theta_u):
    nums = len(data_all)
    dict_column = {}
    for column in list(data_all.columns):
        dict_column[column] = (len(data_all[column][data_all[column] == 0])/nums, len(list(data_all[column].unique())))
    selected_attributes = []
    for column_temp in dict_column.keys():
        if dict_column[column_temp][0] < theta_m and dict_column[column_temp][1] < theta_u:
            selected_attributes.append(column_temp)
    return selected_attributes


def main():
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = data_loaded()
    data_all = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    data_all, columns_value, columns_embed, bag_size = data_preprocess(data_all)
    selected_attributes = cluster_analysis(data_all, opt.theta_m, opt.theta_u)

    clusters = data_cluster(data_all, selected_attributes)
    val_best = training_model_classification(data_all, clusters, columns_value, columns_embed, bag_size, f)

    f.close()


if __name__ == '__main__':
    main()
