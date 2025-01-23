import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer


def set_seed(seed=1439):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextImageDataset(Dataset):
    def __init__(self, csv_file, data_dir, tokenizer, label_map={'negative':0, 'neutral':1, 'positive':2}, 
                 transform=None, max_length=128):
        """
        tokenizer: 选用的 tokenizer，实验里会选用 BERTweetTokenizer
        max_length: BERT 截断/填充的最大长度
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.transform = transform
        self.max_length = max_length
        
        self.samples = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                guid = row['guid']
                tag = row['tag'].strip()
                self.samples.append((guid, tag))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        guid, tag = self.samples[idx]
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read().strip()
        
        encoding = self.tokenizer(
            text, 
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if tag not in self.label_map:
            label = -1
        else:
            label = self.label_map[tag]
        
        return input_ids, attention_mask, img, label

def collate_fn(batch):
    """
    填充 batch 内的数据，使其对齐。
    """
    input_ids_list = []
    attention_mask_list = []
    img_list = []
    label_list = []
    
    for (input_ids, attention_mask, img, lbl) in batch:
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        img_list.append(img)
        label_list.append(lbl)
    
    input_ids = torch.stack(input_ids_list, dim=0)
    attention_mask = torch.stack(attention_mask_list, dim=0)
    images = torch.stack(img_list, dim=0)
    labels = torch.LongTensor(label_list)
    
    return input_ids, attention_mask, images, labels


class BERTweetTextOnly(nn.Module):
    """
    只使用文本的模型
    """
    def __init__(self, num_classes=3, hidden_dim=128, dropout_p=0.4):
        super(BERTweetTextOnly, self).__init__()
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        for param in self.bertweet.parameters():
            param.requires_grad = False
        
        self.text_fc = nn.Linear(768, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, input_ids, attention_mask, images=None):
        """
        只关心文本特征:
          input_ids: (B, L)
          attention_mask: (B, L)
          images: 忽略
        """
        with torch.no_grad():
            bert_outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_outputs.last_hidden_state
        text_feat = torch.mean(last_hidden_state, dim=1)
        text_feat = self.dropout(F.relu(self.text_fc(text_feat)))
        logits = self.classifier(text_feat)
        
        return logits


class ResNet18ImageOnly(nn.Module):
    """
    只使用图像的模型
    """
    def __init__(self, num_classes=3, hidden_dim=128, dropout_p=0.4):
        super(ResNet18ImageOnly, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.img_fc = nn.Linear(512, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, input_ids=None, attention_mask=None, images=None):
        """
        只关心图像特征:
          images: (B, 3, H, W)
        """
        with torch.no_grad():
            feat = self.resnet_backbone(images)
        feat = feat.view(feat.size(0), -1)
        feat = self.dropout(F.relu(self.img_fc(feat)))
        logits = self.classifier(feat)
        
        return logits


class BERTweetFusionResNet18Attn(nn.Module):
    """
    多模态模型 (文本 + 图像), 使用 multi-head attention 做简单融合
    """
    def __init__(self,
                 num_classes=3,
                 hidden_dim=128,
                 nhead=4,
                 dropout_p=0.4):
        super(BERTweetFusionResNet18Attn, self).__init__()
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
        with torch.no_grad():
            bert_outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
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


def train_model(model, train_loader, val_loader, device, epochs=5, lr=4e-4):
    """
    通用的训练函数，可传入任意上述模型 (text-only, image-only, multimodal)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, images, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_model(val_loader, model, device, criterion)
        train_losses.append(train_epoch_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_acc, train_losses, val_losses


def evaluate_model(loader, model, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, images, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc

if __name__ == "__main__":
    set_seed(1439)

    csv_file = "../data/train.txt"
    data_dir = "../data/dataset"
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_img = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = TextImageDataset(
        csv_file=csv_file,
        data_dir=data_dir,
        tokenizer=tokenizer,
        label_map={'negative': 0, 'neutral': 1, 'positive': 2},
        transform=transform_img,
        max_length=64
    )

    dataset_size = len(full_dataset)
    val_size = int(0.05 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1439)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    # === 训练只使用文本的模型 ===
    text_only_model = BERTweetTextOnly(num_classes=3, hidden_dim=128, dropout_p=0.4).to(device)
    print("\n=== 训练 Text-Only 模型 ===")
    text_only_model, best_text_acc, _, _ = train_model(text_only_model, train_loader, val_loader, device)

    # === 训练只使用图像的模型 ===
    image_only_model = ResNet18ImageOnly(num_classes=3, hidden_dim=128, dropout_p=0.4).to(device)
    print("\n=== 训练 Image-Only 模型 ===")
    image_only_model, best_image_acc, _, _ = train_model(image_only_model, train_loader, val_loader, device)

    # === 训练多模态模型 ===
    multimodal_model = BERTweetFusionResNet18Attn(
        num_classes=3, hidden_dim=128, nhead=4, dropout_p=0.4
    ).to(device)
    print("\n=== 训练 MultiModal 模型 ===")
    multimodal_model, best_multimodal_acc, _, _ = train_model(multimodal_model, train_loader, val_loader, device)

    # === 最终评估对比 ===
    acc_values = [best_text_acc, best_image_acc, best_multimodal_acc]
    labels = ["Text Only", "Image Only", "MultiModal"]
    print("\n=== Ablation Plus 实验结果对比 ===")
    for l, acc in zip(labels, acc_values):
        print(f"{l} Val Acc: {acc:.4f}")

    # === 画图对比 ===
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, acc_values, color=['tab:green', 'tab:red', 'tab:blue'])
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height+0.01, f"{height:.2f}", ha='center', fontweight='bold')
    plt.title('Ablation Experiment Plus')
    plt.ylabel('Validation Accuracy')
    plt.savefig("ablation_plus_compare.png")
    plt.show()
