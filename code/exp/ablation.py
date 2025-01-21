import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import random
import numpy as np

from train.train import evaluate_model
from data.bert import TextImageDataset, collate_fn
from model.resnet18_bert_attn import BERTweetFusionResNet18_Attn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model_ablation(loader, model, device, criterion, mode='text'):
    """
    消融实验：只输入文本 或 只输入图像 模式下的验证/测试。
    - mode='text': 只输入文本
    - mode='image': 只输入图像
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
            
            if mode == 'text':
                outputs = model(text_indices, text_lengths, None)
            elif mode == 'image':
                outputs = model(None, None, images)
            else:
                raise ValueError("mode should be 'text' or 'image'.")
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    acc = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, acc

def run_ablation_experiment(
    model, 
    device, 
    val_loader, 
    best_model_path='../best_model.pth', 
    ablation_save_path='ablation.png'
):
    """
    使用指定模型做一次消融实验，并绘制柱状图。
    """
    if not os.path.exists(best_model_path):
        print(f"[Error] Best model path not found: {best_model_path}")
        return
    print("=== Ablation Experiment on the Best Model ===")
    best_model_sd = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(best_model_sd)
    criterion = nn.CrossEntropyLoss()
    _, val_acc_full = evaluate_model(val_loader, model, device, criterion)
    _, val_acc_text = evaluate_model_ablation(val_loader, model, device, criterion, mode='text')
    _, val_acc_image = evaluate_model_ablation(val_loader, model, device, criterion, mode='image')
    print(f"Full Input Val Acc:   {val_acc_full:.4f}")
    print(f"Text-Only Val Acc:    {val_acc_text:.4f}")
    print(f"Image-Only Val Acc:   {val_acc_image:.4f}")

    plt.clf()
    x_labels = ['Full Input', 'Text Only', 'Image Only']
    acc_values = [val_acc_full, val_acc_text, val_acc_image]
    plt.bar(x_labels, acc_values, color=['tab:blue','tab:green','tab:red'])
    plt.ylim(0, 1)
    for i, v in enumerate(acc_values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
    plt.title('Ablation Experiment')
    plt.ylabel('Accuracy')

    ablation_dir = os.path.dirname(ablation_save_path)
    if ablation_dir and not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir, exist_ok=True)
    plt.savefig(ablation_save_path)




if __name__ == '__main__':
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
        batch_size=512,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model = BERTweetFusionResNet18_Attn().to(device)
    run_ablation_experiment(model, device, val_loader, best_model_path='best_model.pth', ablation_save_path='ablation.png')