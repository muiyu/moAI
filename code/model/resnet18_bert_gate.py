import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel


class BERTweetFusionResNet18_Gate(nn.Module):
    def __init__(self, 
                 num_classes=3, 
                 hidden_dim=128,
                 dropout_p=0.4):
        super(BERTweetFusionResNet18_Gate, self).__init__()
        self.hidden_dim = hidden_dim

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        for param in self.bertweet.parameters():
            param.requires_grad = False
        self.text_fc = nn.Linear(768, self.hidden_dim)
        self.text_dropout = nn.Dropout(dropout_p)
        
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False  # 冻结 ResNet18
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.img_fc = nn.Linear(512, self.hidden_dim)
        self.img_dropout = nn.Dropout(dropout_p)
        
        self.gate_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.gate_activation = nn.Sigmoid()
        self.fusion_dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, images):
        with torch.no_grad():
            bert_outputs = self.bertweet(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
        text_seq = self.text_dropout(F.relu(self.text_fc(bert_outputs.last_hidden_state)))
        text_feat = text_seq.mean(dim=1)
        
        with torch.no_grad():
            resnet_output = self.resnet(images)
        resnet_output = resnet_output.view(resnet_output.size(0), -1)
        img_feat = self.img_dropout(F.relu(self.img_fc(resnet_output)))
        
        combined = torch.cat((text_feat, img_feat), dim=1)
        gate = self.gate_activation(self.gate_fc(combined))
        fused_feat = gate * text_feat + (1 - gate) * img_feat
        fused_feat = self.fusion_dropout(fused_feat)
        logits = self.classifier(fused_feat)
        
        return logits
