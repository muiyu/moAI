import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image

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
