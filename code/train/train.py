import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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


def train_model(
    train_loader,
    val_loader,
    model,
    device,
    num_epochs=5,
    lr=4e-4,
    save_path='plot.png',
    best_model_path='best_model.pth'
):
    """
    训练循环，并在每个 epoch 结束后在验证集上评估 loss 和 acc。
    使用 history 变量存储训练和验证信息，并基于 history 生成并保存图表。
    训练并保存最优模型。
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'steps': [],
        'val_steps': []
    }
    
    total_steps = 0
    best_val_acc = 0.0
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_total_correct = 0
        epoch_total_samples = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
            for step, batch in enumerate(train_loader, start=1):
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
                preds = outputs.argmax(dim=1)
                epoch_total_correct += (preds == labels).sum().item()
                epoch_total_samples += len(labels)

                history['train_loss'].append(loss.item())
                history['steps'].append(total_steps)

                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                pbar.update(1)

        train_avg_loss = epoch_total_loss / epoch_total_samples
        train_acc = epoch_total_correct / epoch_total_samples
        history['train_acc'].append(train_acc)
        val_loss, val_acc = evaluate_model(val_loader, model, device, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_steps'].append(total_steps)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_acc={best_val_acc:.4f} at {best_model_path}")

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Avg Loss: {train_avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        plt.clf()
        if history['steps']:
            sampled_indices = range(0, len(history['train_loss']), 30)
            sampled_steps = [history['steps'][i] for i in sampled_indices]
            sampled_train_losses = [history['train_loss'][i] for i in sampled_indices]
            plt.plot(sampled_steps, sampled_train_losses, label='Train Loss', 
                     color='tab:red', marker='o', linestyle='-', markersize=4)
        if history['val_loss']:
            plt.plot(history['val_steps'], history['val_loss'], label='Validation Loss', 
                     color='tab:orange', marker='x', linestyle='--', markersize=6)
        
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.xlim(0, total_steps + 1)
        all_losses = history['train_loss'] + history['val_loss']
        if all_losses:
            plt.ylim(0, max(all_losses) * 1.1)
        
        plt.savefig(save_path)

    print("\nTraining complete.")
    print(f"Best model was saved with val_acc={best_val_acc:.4f} at {best_model_path}")