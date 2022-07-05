import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, visualbert_model, tokenizer, dropout=0.25):
        super(Model, self).__init__()
        Ks = [3, 4, 5]
        in_channel = 1
        out_channel = 100
        embedding_dim = 768
        self.bert = visualbert_model
        self.tokenizer = tokenizer
        self.imghead = nn.Sequential(nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Dropout(p=0.1))
        self.conv1 = nn.Conv1d(1, 5, 3, padding=1)       # [(n+2p-f) / s + 1]    2048 + 2p - 4 +1
        self.conv2 = nn.Conv1d(1, 5, 9, padding=4)
        self.conv3 = nn.Conv1d(1, 5, 27, padding=13)
        self.conv4 = nn.Conv1d(1, 5, 91, padding=45)
        self.conv5 = nn.Conv1d(1, 5, 273, padding=136)         # consider other kernel size
        self.conv_final = nn.Conv1d(25, 5, 3, padding=1)
        self.device = torch.device('cuda')
        self.criterion = torch.nn.MultiLabelSoftMarginLoss()
        self.classify = nn.Sequential(
            #
            # nn.Linear(556, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(),
            # nn.Dropout(p=dropout),
            #
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),

            nn.Linear(512, 13)
        )

    def forward(self, text, feature, labels=None, masks=None):
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=40).to(self.device)
        feature = feature.float().to(self.device).unsqueeze(1)       #
        v1 = self.conv1(feature)
        v2 = self.conv2(feature)
        v3 = self.conv3(feature)
        v4 = self.conv4(feature)
        v5 = self.conv5(feature)
        visual_embeds = torch.cat([v1, v2, v3, v4 ,v5], 1)
        visual_embeds = self.conv_final(visual_embeds)


        visual_embeds = self.imghead(visual_embeds)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(self.device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(self.device)
        output_hidden_states = True
        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
                "output_hidden_states": output_hidden_states
            }
        )
        outputs = self.bert(**inputs)
        last_hidden_states = outputs.hidden_states[12][:, 0, :]
        outputs = self.classify(last_hidden_states)
        # last_hidden_states = outputs.hidden_states[12][:,0,:]      # batchsize * 45 * 768
        # conv

        # outputs = self.classify(last_hidden_states)
        if labels != None:
            label = labels.float().to(self.device)
            loss = self.criterion(outputs, label)  # + sigmoid
            return loss, outputs
        else:
            return outputs


if __name__ == "__main__":
    pass
