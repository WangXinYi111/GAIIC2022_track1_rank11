from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import jieba
import torch
from transformers import BertTokenizer
from transformers import BertPreTrainedModel, BertModel

class textDataset(Dataset):
    def __init__(self, tokenizer, label_dir='../data/label.json', feature_dir='../data/feature_imgName.json',
                 index = None):
        super().__init__()
        with open(label_dir, 'r') as f:
            lable_title = json.load(f)
        self.tokenizer = tokenizer
        self.titles = []
        self.labels = []
        self.img_names = []
        for i in index:
            self.titles.append(lable_title['title'][i])
            self.labels.append(lable_title['label'][i])
            self.img_names.append(lable_title['img_name'][i])
        # ------all data-------
        # self.titles_all=lable_title['title']
        # self.labels_all=lable_title['label']
        # self.img_names_all=lable_title['image_name']
        with open(feature_dir, 'r') as f:
            self.img_features = json.load(f)
        #

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.titles[idx]
        name = self.img_names[idx]
        feature = np.array(self.img_features[name])
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=40,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )
        #
        return encoding['input_ids'][0], encoding["attention_mask"], np.array(label), feature   # why attention_mask have 3 dimension?

#
if __name__ == "__main__":
    #
    tokenizer = BertTokenizer.from_pretrained("/home1/wxwHD/GAIIC_track1_baseline/Bert_wwm/chinese_wwm_pytorch")
    model = BertModel.from_pretrained("/home1/wxwHD/GAIIC_track1_baseline/Bert_wwm/chinese_wwm_pytorch")

    with open('../data/word_to_idx_v1.json', 'r') as f:
        word_to_idx = json.load(f)
    index = [i for i in range(10)]
    train_dataset = textDataset(tokenizer, label_dir ='../data/label_v1.json',
                                feature_dir ='../data/feature_imgName.json',
                                index = index)
    trainloader = DataLoader(train_dataset,
                             batch_size = 5,
                             shuffle = False,
                             num_workers = 0)
    for data in trainloader:
        inputs, attention, labels, feature = data
        print(inputs)
        print(attention)
        print(labels)
        print(feature)
        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        # print(inputs.shape, labels.shape, feature.shape) # torch.Size([2, 30]) torch.Size([2, 13]) torch.Size([2, 2048])
