import numpy as np
#from gensim.models import word2vec
from sklearn import metrics
import os
from sklearn.model_selection import GroupKFold
from bert_net import Model
from src.dataset_Bert import textDataset
import torch
from torch.utils.data import DataLoader
import time
import warnings
warnings.filterwarnings("ignore")
import random
import json
from transformers import BertTokenizer
from transformers import BertModel

random.seed(2022)


# label_dir = "/home1/wxwHD/GAIIC_track1_baseline/New_baseline/data/label_v1_1.json"
# feature_dir = "/home1/wxwHD/GAIIC_track1_baseline/New_baseline/data/feature_imgName_1.json"
label_dir = "data/tmp_data/label_v1.json"
feature_dir = "data/tmp_data/feature_imgName_v1.json"
# weight_path = "/home1/wxwHD/GAIIC_track1_baseline/New_baseline/ckpt_v28_new_model/fold_1_best.pth"


with open(label_dir, 'r') as f:
    groups = json.load(f)['img_name']

def train_model(model, criterion, optimizer, lr_scheduler=None):
    total_iters = len(trainloader)    # why this fun can see "trainloader"
    print('total_iters:{}'.format(total_iters))
    since = time.time()
    best_loss = 1e7
    best_epoch = 0
    #
    iters = len(trainloader)
    for epoch in range(1, max_epoch + 1):
        model.train(True)
        begin_time = time.time()
        print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        print('Fold{} Epoch {}/{}'.format(fold+1, epoch, max_epoch))
        print('-' * 10)
        count = 0
        train_loss = []
        for i, (inputs, attention, labels, frt) in enumerate(trainloader):
            count += 1
            inputs = inputs.type(torch.LongTensor).to(device)
            labels = labels.to(device).float()
            frt = frt.to(device).float()
            attention = attention.type(torch.LongTensor).to(device)
            #
            out_linear = model(inputs, attention, frt)
            loss = criterion(out_linear, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新cosine学习率
            if lr_scheduler != None:
                lr_scheduler.step(epoch + count / iters)
            if print_interval > 0 and (i % print_interval == 0 or out_linear.size()[0] < train_batch_size):
                spend_time = time.time() - begin_time
                print(
                    ' Fold:{} Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        fold+1, epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
            #
            train_loss.append(loss.item())
        #lr_scheduler.step()
        val_auc, val_loss= val_model(model, criterion)
        print('valLogLoss: {:.4f} valAuc: {:.4f}'.format(val_loss, val_auc))
        model_out_path = model_save_dir + "/" + 'fold_' + str(fold+1) + '_'+str(epoch) + '.pth'
        best_model_out_path = model_save_dir + "/" + 'fold_' + str(fold+1) + '_best' + '.pth'
        model_out_path8 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '8' + '.pth'
        model_out_path10 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '10' + '.pth'
        model_out_path12 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '12' + '.pth'
        model_out_path16 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '16' + '.pth'
        model_out_path20 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '20' + '.pth'
        model_out_path24 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '24' + '.pth'


        #save the best model
        if (val_loss < best_loss) :
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_out_path)
            print("save best epoch: {} best auc: {} best logloss: {}".format(best_epoch, val_auc, val_loss))

        if epoch == 8:
            torch.save(model.state_dict(), model_out_path8)
            print("save epoch 8: {} best auc: {} logloss: {}".format(epoch, val_auc, val_loss))
        if epoch == 10:
            torch.save(model.state_dict(), model_out_path10)
            print("save epoch 10: {} best auc: {} logloss: {}".format(epoch, val_auc, val_loss))
        if epoch == 12:
            torch.save(model.state_dict(), model_out_path12)
            print("save epoch 12: {} best auc: {} logloss: {}".format(epoch, val_auc, val_loss))
        if epoch == 16:
            torch.save(model.state_dict(), model_out_path16)
            print("save epoch 16: {} best auc: {} logloss: {}".format(epoch, val_auc, val_loss))
        if epoch == 20:
            torch.save(model.state_dict(), model_out_path20)
            print("save epoch 20: {} best auc: {} logloss: {}".format(epoch, val_auc, val_loss))
        if epoch == 24:
            torch.save(model.state_dict(), model_out_path24)
            print("save epoch 24: {} best auc: {} logloss: {}".format(epoch, val_auc, val_loss))



    print('Fold{} Best logloss: {:.3f} Best epoch:{}'.format(fold+1, best_loss, best_epoch))
    time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return best_loss

@torch.no_grad()
def val_model(model, criterion):
    dset_sizes=len(val_dataset)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pres_list = []
    labels_list = []
    for data in val_loader:
        inputs, attention, labels, frt = data
        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        inputs, attention ,labels, frt = inputs.cuda(), attention.cuda(), labels.cuda(), frt.cuda().float()
        outputs = model(inputs, attention, frt)
        #pres_list.append(outputs.sigmoid().detach().cpu().numpy())
        #labels_list.append(labels.detach().cpu().numpy())
        pres_list += outputs.sigmoid().detach().cpu().numpy().tolist()
        labels_list += labels.detach().cpu().numpy().tolist()
    #
    try:
        val_auc = metrics.roc_auc_score(labels_list, pres_list, multi_class='ovo')
    except ValueError:
        val_auc = 1e2
        pass
    log_loss = metrics.log_loss(labels_list, pres_list)#
    return val_auc, log_loss
#
if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained("data/pretrain_model/chinese_wwm_pytorch")
    bert_model = BertModel.from_pretrained("data/pretrain_model/2nd_pretrain_Bertmodel")

    create_path = 'data/model_data/Bert_model1/'
    isExists = os.path.exists(create_path)
    if not isExists:
        os.makedirs(create_path)

    model_save_dir = 'data/model_data/Bert_model1/'
    print_interval = 100
    train_batch_size = 256
    val_batch_size = 256
    max_epoch = 40
    device = torch.device('cuda')
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    folds = GroupKFold(n_splits = 30).split(np.arange(len(groups)), groups = groups)
    kfold_best = []
    for fold, (trn_idx, val_idx) in enumerate(folds):
        #
        print('train fold: {} len train: {} len val: {}'.format(fold+1, len(trn_idx), len(val_idx)))
        model = Model(bert_model, class_num = 13)
        # model.load_state_dict(torch.load(weight_path))
        model.to(device)                                       # consider decrease lr
        optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-4 , weight_decay = 5e-4)     # weight_decay?
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)      # ready to change gamma
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=5e-6, last_epoch=-1)
        train_dataset = textDataset(tokenizer, label_dir, feature_dir, trn_idx)
        trainloader = DataLoader(train_dataset,
                                batch_size = train_batch_size,
                                shuffle = True,
                                num_workers = 8)
        val_dataset = textDataset(tokenizer, label_dir, feature_dir, val_idx)
        val_loader = DataLoader(val_dataset,
                                batch_size = val_batch_size,
                                shuffle = False,
                                num_workers = 8)
        best_loss = train_model(model, criterion, optimizer, lr_scheduler = lr_scheduler)
        kfold_best.append(best_loss)
    print("local cv:", kfold_best, np.mean(kfold_best))