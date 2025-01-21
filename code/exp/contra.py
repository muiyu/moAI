import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np

from data.bert import TextImageDataset, collate_fn
from model.resnet18_bert_attn import BERTweetFusionResNet18_Attn
from model.resnet18_bert_gate import BERTweetFusionResNet18_Gate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(loader, model, device, criterion):
    """
    在给定的数据加载器上做验证/测试。返回平均loss和准确率。
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            text_indices, text_lengths, images, labels = batch
            text_indices = text_indices.to(device)
            text_lengths = text_lengths.to(device)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(text_indices, text_lengths, images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    acc = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, acc


def train_single_model(
    train_loader, 
    val_loader, 
    model, 
    device, 
    num_epochs=5, 
    lr=1e-3, 
    best_model_path='best_model.pth'
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'val_loss': [],
        'steps': [],
        'val_steps': [],
    }

    best_val_acc = 0.0
    total_steps = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_total_samples = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
            for step, batch in enumerate(train_loader, 1):
                total_steps += 1
                text_indices, text_lengths, images, labels = batch
                text_indices = text_indices.to(device)
                text_lengths = text_lengths.to(device)
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(text_indices, text_lengths, images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_total_loss += loss.item() * len(labels)
                epoch_total_samples += len(labels)

                history['train_loss'].append(loss.item())
                history['steps'].append(total_steps)

                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                pbar.update(1)
        
        val_loss, val_acc = evaluate_model(val_loader, model, device, criterion)
        history['val_loss'].append(val_loss)
        history['val_steps'].append(total_steps)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model for this run saved with val_acc={best_val_acc:.4f} at {best_model_path}")

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Avg Loss: {epoch_total_loss/epoch_total_samples:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history


def contrast_experienment(
    train_loader, 
    val_loader, 
    model1, 
    model2,
    device,
    num_epochs=12, 
    lr=4e-4, 
    save_path='contrast_plot.png',
    best_model_path_1='best_model_1.pth',
    best_model_path_2='best_model_2.pth'
):
    """
    对比实验：
      1. 分别训练 model1 和 model2
      2. 将它们的训练与验证 loss 画到同一张图上，便于对比
    
    参数:
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - model1, model2: 待对比的两个模型
    - device: CPU/GPU
    - num_epochs: 训练轮数
    - lr: 学习率
    - save_path: 保存图像的路径
    - best_model_path_1, best_model_path_2: 分别保存两个模型的最佳参数
    """
    print("==== Start training Model 1 ====")
    history1 = train_single_model(
        train_loader, 
        val_loader, 
        model1, 
        device, 
        num_epochs=num_epochs, 
        lr=lr, 
        best_model_path=best_model_path_1
    )
    print("\n==== Start training Model 2 ====")
    history2 = train_single_model(
        train_loader, 
        val_loader, 
        model2, 
        device, 
        num_epochs=num_epochs, 
        lr=lr, 
        best_model_path=best_model_path_2
    )


    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if history1['steps']:
        sampled_indices_1 = range(0, len(history1['train_loss']), 30)
        sampled_steps_1 = [history1['steps'][i] for i in sampled_indices_1]
        sampled_train_losses_1 = [history1['train_loss'][i] for i in sampled_indices_1]
        plt.plot(sampled_steps_1, sampled_train_losses_1,
                 label='Model1 Train Loss', color='tab:blue', marker='o', linestyle='-')
    if history1['val_steps']:
        plt.plot(history1['val_steps'], history1['val_loss'],
                 label='Model1 Val Loss', color='tab:blue', marker='x', linestyle='--')
    if history2['steps']:
        sampled_indices_2 = range(0, len(history2['train_loss']), 30)
        sampled_steps_2 = [history2['steps'][i] for i in sampled_indices_2]
        sampled_train_losses_2 = [history2['train_loss'][i] for i in sampled_indices_2]
        plt.plot(sampled_steps_2, sampled_train_losses_2,
                 label='Model2 Train Loss', color='tab:red', marker='o', linestyle='-')
    if history2['val_steps']:
        plt.plot(history2['val_steps'], history2['val_loss'],
                 label='Model2 Val Loss', color='tab:red', marker='x', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Contrast Experiment: Training and Validation Loss')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()

    max_step = 0
    if history1['steps']:
        max_step = max(max_step, max(history1['steps']))
    if history2['steps']:
        max_step = max(max_step, max(history2['steps']))
    plt.xlim(0, max_step + 1)
    all_train_losses = history1['train_loss'] + history2['train_loss']
    all_val_losses = history1['val_loss'] + history2['val_loss']
    all_losses = all_train_losses + all_val_losses
    if all_losses:
        plt.ylim(0, max(all_losses) * 1.1)

    plt.savefig(save_path)
    plt.close()
    print("Contrast experiment complete.")




if __name__ == "__main__":
    set_seed(1439)
    
    csv_file = "../data/train.txt"
    data_dir = "../data/dataset" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    
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
        max_length=64  # 根据需要可调
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
        batch_size=512,
        shuffle=False,
        collate_fn=collate_fn
    )
    model1 = BERTweetFusionResNet18_Attn().to(device)
    model2 = BERTweetFusionResNet18_Gate().to(device)
    contrast_experienment(train_loader, val_loader, model1, model2, device, num_epochs=12, lr=4e-4)