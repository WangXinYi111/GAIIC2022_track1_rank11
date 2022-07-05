

# 'Visual_model_freeze_POS_13/'   data: 13(+25000)   lr:1e-5
#  corresponding net file: visual_bert_model_all.py
#  result: test_B_all_freeze_POS_13.txt(not test yet)

# 'Visual_model_freeze_sequence13_POS14/'  data: 14     lr: 1e-5
#  corresponding net file: visual_bert_model_all_var.py(sequence=15)

from transformers import BertTokenizer, VisualBertForVisualReasoning, BertModel
import torch
from torch import nn
from dataset import textDataset, feature_dataset, mask_dataset
import json
import os
from sklearn.model_selection import GroupKFold
import numpy as np
from torch.utils.data import DataLoader
import time
from visual_bert_model_all_var import Model
from sklearn import metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

label_dir = "/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/data/label_v14.json"
feature_dir = "/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/data/feature_imgName_14.json"
weight_path_from_v_m_i = "/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/Visual_model_integration/fold_1_best.pth"
weight_path = "/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/change_embedding_visual_bert.pth"
model_save_dir_pth =  "/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/change_embedding_change_cls_visual_bert.pth"

with open(label_dir, 'r') as f:
    groups = json.load(f)['img_name']

def train_model(model, optimizer, lr_scheduler=None):
    total_iters = len(trainloader)
    print('total_iters:{}'.format(total_iters))
    since = time.time()
    best_loss = 1e7
    best_epoch = 0
    iters = len(trainloader)
    for epoch in range(1, max_epoch + 1):
        model.train(True)
        model.bert.visual_bert.embeddings.position_embeddings.trainable = False
        model.bert.visual_bert.embeddings.position_embeddings._parameters.requires_grad = False
        para_list = []
        for para in model.parameters():
            # print(para)
            if ~para.data.any():
                para.requires_grad = False
            para_list.append(para)
        count = 0
        begin_time = time.time()
        print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        print('Fold{} Epoch {}/{}'.format(fold+1, epoch, max_epoch))
        print('-' * 10)
        train_loss = []
        for i, (text, feature, labels) in enumerate(trainloader):
            count += 1
            loss, outputs = model(text, feature, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if print_interval > 0 and (i % print_interval == 0):
                spend_time = time.time() - begin_time
                print(
                    ' Fold:{} Epoch:{}({}/{}) loss:{:.3f} lr:{:.25f} epoch_Time:{}min:'.format(
                        fold+1, epoch, count, total_iters,
                        loss, optimizer.param_groups[-1]['lr'],              # loss_sum/train_batch_size
                        spend_time / count * total_iters // 60 - spend_time // 60))
                print("output.logits: {}".format(outputs.logits[0,:]))

        lr_scheduler.step()
        loss_sum, acc = val_model(model)
        best_model_out_path = model_save_dir + "/" + 'fold_' + str(fold + 1) + '_best' + '.pth'
        model_out_path3 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '3' + '.pth'
        model_out_path6 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '6' + '.pth'
        model_out_path8 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '8' + '.pth'
        model_out_path10 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '10' + '.pth'
        model_out_path12 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '12' + '.pth'
        model_out_path16 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '16' + '.pth'
        model_out_path20 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '20' + '.pth'
        model_out_path24 = model_save_dir + "/" + 'fold_' + str(fold + 1) + '24' + '.pth'
        best_model_out_path = model_save_dir + "/" + 'fold_' + str(fold + 1) + '_best' + '.pth'
        if (loss_sum < best_loss) :
            best_loss = loss_sum
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_out_path)
            print("save best epoch: {} best lossum: {} acc :{}".format(best_epoch, best_loss, acc))
        # print(loss_sum)
        if epoch == 3:
            torch.save(model.state_dict(), model_out_path3)
            print("save epoch 3: {} best lossum: {} acc :{}".format(best_epoch, best_loss, acc))
        if epoch == 6:
            torch.save(model.state_dict(), model_out_path6)
            print("save epoch 6:  {} best lossum: {} acc :{}".format(best_epoch, best_loss, acc))
        if epoch == 8:
            torch.save(model.state_dict(), model_out_path8)
            print("save epoch 8: {} best lossum: {} acc :{}".format(best_epoch, best_loss, acc))
        if epoch == 10:
            torch.save(model.state_dict(), model_out_path10)
            print("save epoch 10:  {} best lossum: {} acc :{}".format(best_epoch, best_loss, acc))
        if epoch == 12:
            torch.save(model.state_dict(), model_out_path12)
            print("save epoch 12:  {} best lossum: {} acc :{}".format(best_epoch, best_loss, acc))
        if epoch == 16:
            torch.save(model.state_dict(), model_out_path16)
            print("save epoch 16:  {} best lossum: {} acc :{}".format(best_epoch, best_loss, acc))
        if epoch == 20:
            torch.save(model.state_dict(), model_out_path20)
            print("save epoch 20: {} best lossum:{} acc :{}".format(best_epoch, best_loss, acc))
        if epoch == 24:
            torch.save(model.state_dict(), model_out_path24)
            print("save epoch 24:  {} best lossum:{} acc :{}".format(best_epoch, best_loss, acc))

    print('Fold{} Best logloss: {:.3f} Best epoch:{}'.format(fold + 1, best_loss, best_epoch))
    return best_loss

@torch.no_grad()
def val_model(model):
    dset_sizes=len(val_dataset)
    model.eval()
    label_logits = []
    label_compare = []
    loss_sum = 0
    for data in val_loader:
        text, feature, labels = data
        loss, outputs = model(text, feature, labels)
        for i in outputs.logits.reshape(-1):
            label_logits.append(1) if i > 0 else label_logits.append(0)
        for j in labels.reshape(-1):
            label_compare.append(1) if j == 1 else label_compare.append(0)
        loss_sum += loss
    num_right = 0
    num_total = len(label_compare)
    for i in range(len(label_compare)):
        if label_logits[i] == label_compare[i]:
            num_right += 1
    acc =  num_right/ num_total

    return loss_sum, acc
#

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("/home1/wxwHD/GAIIC_track1_baseline/Bert_wwm/chinese_wwm_pytorch")
    # visualBert_model = torch.load(weight_path)
    # pretrained_dict_singlemodel = visualBert_model.state_dict()
    #
    # model = Model(visualBert_model, tokenizer)
    # model.load_state_dict(torch.load(weight_path_from_v_m_i))
    # pretrained_dict = model.state_dict()
    #
    # for i in pretrained_dict.keys():
    #     for j in pretrained_dict_singlemodel.keys():
    #         if ("bert."+ j) == i :
    #             pretrained_dict_singlemodel[j] = pretrained_dict[i]
    #
    #
    #
    # torch.save(visualBert_model, model_save_dir_pth)
    # model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")
    # ori_dict = model.state_dict()
    # visualBert_model.load_state_dict(pretrained_dict_singlemodel)
    # a = nn.Linear(768, 13)
    # visualBert_model.cls = a

    visualBert_model = torch.load(model_save_dir_pth)
    dict = visualBert_model.state_dict()
    b = nn.Embedding(512,768)
    torch.nn.init.zeros_(b.weight)
    visualBert_model.visual_bert.embeddings.position_embeddings = b
    visualBert_model.visual_bert.embeddings.position_embeddings.trainable = False
    visualBert_model.visual_bert.embeddings.position_embeddings.requires_grad = False
    dict1 = visualBert_model.state_dict()
    # # visualBert_model.load_state_dict(torch.load('/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/Visual_model/fold_1_best.pth'))
    #
    # visualBert_model_now  = torch.load("/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/Visual_model_temp/fold_1_best.pth")

    model_save_dir = 'Visual_model_freeze_sequence13_POS14/'
    model = Model(visualBert_model, tokenizer)
    # model.load_state_dict(torch.load("/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/Visual_model_freeze_POS_13/fold_112.pth"))
    print_interval = 100
    train_batch_size = 200
    val_batch_size = 200
    max_epoch = 50
    device = torch.device('cuda')
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    folds = GroupKFold(n_splits=30).split(np.arange(len(groups)), groups=groups)
    kfold_best = []
    for fold, (trn_idx, val_idx) in enumerate(folds):
        #
        print('train fold: {} len train: {} len val: {}'.format(fold + 1, len(trn_idx), len(val_idx)))
        model.to(device)  # consider decrease lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-4)  # weight_decay?
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # ready to change gamma

        train_dataset = mask_dataset(tokenizer, label_dir, feature_dir, trn_idx)
        trainloader = DataLoader(train_dataset,
                                 batch_size=train_batch_size,
                                 shuffle=True,
                                 num_workers=8,
                                 drop_last=True)
        val_dataset = mask_dataset(tokenizer, label_dir, feature_dir, val_idx)
        val_loader = DataLoader(val_dataset,
                                batch_size=val_batch_size,
                                shuffle=False,
                                num_workers=8,
                                drop_last=True)
        best_loss = train_model(model, optimizer, lr_scheduler=lr_scheduler)
        kfold_best.append(best_loss)

    print("local cv:", kfold_best, np.mean(kfold_best))
