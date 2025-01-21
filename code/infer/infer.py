import os

import torch
import csv
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from data.bert import TextImageDataset, collate_fn
from model.resnet18_bert_attn import BERTweetFusionResNet18_Attn

def main():
    best_model_path = 'best_model.pth'
    test_csv = '../data/test_without_label.txt'
    data_dir = '../data/dataset'
    output_csv = 'results.txt'
    
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTweetFusionResNet18_Attn()
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    test_dataset = TextImageDataset(
        csv_file=test_csv,
        data_dir=data_dir,
        tokenizer=tokenizer,
        transform=test_transform,
        label_map=label_map
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    all_predictions = []
    global_index = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, images, _ = batch
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            
            outputs = model(input_ids, attention_mask, images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            batch_size = preds.shape[0]
            for i in range(batch_size):
                guid, _ = test_dataset.samples[global_index]
                pred_idx = preds[i]
                pred_label_str = inv_label_map[pred_idx]
                
                all_predictions.append((guid, pred_label_str))
                global_index += 1
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['guid', 'tag'])
        for guid, pred in all_predictions:
            writer.writerow([guid, pred])


if __name__ == "__main__":
    main()
