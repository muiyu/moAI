import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel


class BERTweetFusionResNet18_Attn(nn.Module):
    '''
    采用多头注意力融合的文本-图像融合模型。

    1. 文本编码器：BERTweet + FC
    2. 图像编码器：ResNet18 + FC
    3. 多头注意力融合模块，将图像特征作为 query，与文本序列特征做 attention
    4. 分类层
    '''
    def __init__(self, 
                 num_classes=3, 
                 hidden_dim=128,
                 nhead=4,
                 dropout_p=0.4):
        super(BERTweetFusionResNet18_Attn, self).__init__()
        self.hidden_dim = hidden_dim

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        for param in self.bertweet.parameters():
            param.requires_grad = False
        self.text_fc = nn.Linear(768, self.hidden_dim)
        self.text_dropout = nn.Dropout(dropout_p)
        
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.img_fc = nn.Linear(512, self.hidden_dim)
        self.img_dropout = nn.Dropout(dropout_p)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=nhead, 
            dropout=dropout_p,
            batch_first=True
        )
        
        self.fusion_dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)


    def forward(self, input_ids, attention_mask, images):
        # ----------- 只输入文本的情况 -----------
        if input_ids is not None and images is None:
            with torch.no_grad():
                bert_outputs = self.bertweet(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
            text_seq = self.text_dropout(F.relu(self.text_fc(bert_outputs.last_hidden_state)))
            fused_token = text_seq.mean(dim=1)
            fused_token = self.fusion_dropout(fused_token)
            logits = self.classifier(fused_token)
            
            return logits

        # ----------- 只输入图像的情况 -----------
        elif input_ids is None and images is not None:
            with torch.no_grad():
                resnet_output = self.resnet(images)
            resnet_output = resnet_output.view(resnet_output.size(0), -1)
            img_feat = self.img_dropout(F.relu(self.img_fc(resnet_output)))
            fused_token = self.fusion_dropout(img_feat)
            logits = self.classifier(fused_token)
            
            return logits
        
        # ----------- 文本+图像同时输入 -----------
        else:
            with torch.no_grad():
                bert_outputs = self.bertweet(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
            text_seq = self.text_dropout(F.relu(self.text_fc(bert_outputs.last_hidden_state)))
            
            with torch.no_grad():
                resnet_output = self.resnet(images)
            resnet_output = resnet_output.view(resnet_output.size(0), -1)
            img_feat = self.img_dropout(F.relu(self.img_fc(resnet_output)))
            
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.bool()
            else:
                key_padding_mask = None
            attn_output, attn_weights = self.multihead_attn(
                query=img_feat.unsqueeze(1),
                key=text_seq,
                value=text_seq,
                key_padding_mask=key_padding_mask
            )
            fused_token = attn_output.squeeze(1)
            
            fused_token = self.fusion_dropout(fused_token)
            logits = self.classifier(fused_token)
            
            return logits
