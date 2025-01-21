import os

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
import argparse
import torchvision.transforms as transforms
from transformers import AutoTokenizer

from data.bert import TextImageDataset, collate_fn
from model.resnet18_bert_attn import BERTweetFusionResNet18_Attn
from train.train import train_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(1439)
    
    parser = argparse.ArgumentParser(description="Training Parameters")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    args = parser.parse_args()
    
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
    
    model = BERTweetFusionResNet18_Attn(num_classes=3, hidden_dim=128, dropout_p=0.4).to(device)
    train_model(train_loader, val_loader, model, device, num_epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
