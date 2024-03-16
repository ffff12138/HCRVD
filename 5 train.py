import pandas as pd
import random
import torch
import time
import numpy as np
import argparse
from gensim.models.word2vec import Word2Vec
from model import TGGA
from torch.autograd import Variable
import lap
import pickle
import torch.nn as nn
from torch.nn import functional as F
from prettytable import PrettyTable
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix

def parse_options():
    parser = argparse.ArgumentParser(description='Training.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./data/pdgs')
    parser.add_argument('-o', '--output', help='The dir path of output', type=str, default='./output')
    args = parser.parse_args()
    return args

def sava_data(filename, data):
    print("Begin to save data：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def get_accuracy(labels, prediction):
    cm = confusion_matrix(labels, prediction)
    def linear_assignment(cost_matrix):
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy

def get_MCM_score(y_true, predictions):
    accuracy = get_accuracy(y_true, predictions)
    MCM = multilabel_confusion_matrix(y_true, predictions)

    tn = MCM[:, 0, 0]
    fp = MCM[:, 0, 1]
    fn = MCM[:, 1, 0]
    tp = MCM[:, 1, 1]
    fpr_array = fp / (fp + tn)
    fnr_array = fn / (tp + fn)
    f1_array = 2 * tp / (2 * tp + fp + fn)
    sum_array = fn + tp
    M_fpr = fpr_array.mean()
    M_fnr = fnr_array.mean()
    M_f1 = f1_array.mean()
    W_fpr = (fpr_array * sum_array).sum() / sum( sum_array )
    W_fnr = (fnr_array * sum_array).sum() / sum( sum_array )
    W_f1 = (f1_array * sum_array).sum() / sum( sum_array )
    return {
        "M_fpr": format(M_fpr * 100, '.3f'),
        "M_fnr": format(M_fnr * 100, '.3f'),
        "M_f1" : format(M_f1 * 100, '.3f'),
        "W_fpr": format(W_fpr * 100, '.3f'),
        "W_fnr": format(W_fnr * 100, '.3f'),
        "W_f1" : format(W_f1 * 100, '.3f'),
        "ACC"  : format(accuracy * 100, '.4f'),
        "MCM" : MCM
    }

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels ,adj= [], [],[]
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2])
        adj.append(item[3])
    return data, torch.LongTensor(labels),adj

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = parse_options()
    input_path = args.input
    output_path = args.output
    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'
    if output_path[-1] == '/':
        output_path = output_path
    else:
        output_path += '/'

    root = input_path
    result_save_path=output_path+'result/'
    model_save_path=output_path+'model/'
    train_data = pd.read_pickle(root+'train/CPs.pkl')
    val_data = pd.read_pickle(root + 'dev/CPs.pkl')
    test_data = pd.read_pickle(root+'test/CPs.pkl')

    setup_seed(42)
    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings=None
    ENCODE_DIM = 128
    EPOCHS = 100
    BATCH_SIZE = 64
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    model = TGGA(EMBEDDING_DIM,MAX_TOKENS+1,ENCODE_DIM,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.to("cuda:0")

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters,weight_decay=1e-8)
    loss_function = torch.nn.BCELoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model
    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    learning_record_dict = {}
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        pre = []
        label = []
        while i < len(train_data):
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels ,train_adj= batch
            if USE_GPU:
                train_inputs, train_labels,train_adj = train_inputs, train_labels.to("cuda:0"),train_adj

            model.zero_grad()
            model.batch_size = len(train_labels)
            output = model(train_inputs,train_adj)

            loss = loss_function(output, Variable(train_labels).to(torch.float32))
            loss.backward()
            optimizer.step()

            predicted=torch.where(output.data > 0.5, 1, 0)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()*len(train_inputs)
            pre += list(np.array(predicted.flatten().cpu()))
            label += list(np.array(train_labels.cpu()))

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        train_score_dict = get_MCM_score(label, pre)

        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        pre=[]
        label=[]
        while i < len(val_data):
            batch = get_batch(val_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            val_inputs, val_labels ,val_adj= batch
            if USE_GPU:
                val_inputs, val_labels ,val_adj= val_inputs, val_labels.to("cuda:0"),val_adj

            model.batch_size = len(val_labels)
            output = model(val_inputs,val_adj)
            loss = loss_function(output, Variable(val_labels).to(torch.float32))

            predicted = torch.where(output.data > 0.5, 1, 0)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item()*len(val_inputs)
            pre += list(np.array(predicted.flatten().cpu()))
            label += list(np.array(val_labels.cpu()))
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        val_score_dict = get_MCM_score(label, pre)
        end_time = time.time()

        if total_acc/total > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.4f, Validation Acc: %.4f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))
        test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        test_table.add_row(
            ["val", str(epoch + 1), format(val_loss_[-1], '.4f')] + [val_score_dict[j] for j in val_score_dict if j != "MCM"])
        # print(test_table)
        learning_record_dict[epoch] = {'train_loss': train_loss_[epoch], 'val_loss': val_loss_[epoch], \
                                       "train_score": train_score_dict, "val_score": val_score_dict}
    sava_data(result_save_path+'result', learning_record_dict)

    model = best_model
    torch.save(model.state_dict(), model_save_path+'TGGA')



if __name__ == '__main__':
    main()
    #python 5\ train.py -i ./data/pdgs/ -o ./output/
