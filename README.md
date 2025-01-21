# Lab5（期末结课项目）
此仓库为本人修读课程《当代人工智能》的 Lab5 代码仓库，包含实验报告内涉及的所有代码。

本实验将尝试构建模型以拟合一个多模态情感分类数据集。具体的实验内容以及说明您应当查看实验报告，本仓库仅包含相应代码。

## Structure
```
|----code
    |----data                       # 数据预处理的代码
    |----exp                        # 消融实验和比较实验使用的代码
    |----model                      # 模型代码
    |----train                      # 训练代码
    |----main.py                    # 主程序，执行模型训练
|----data                           # 数据集目录
    |----dataset                    # 包含所有文本和图像数据
    |----test_without_label.txt     # 测试集文件，用于预测
    |----train.txt                  # 训练数据，包含样本 guid 和对应的标签
|----.gitignore
|----LICENSE
|----README.md
```

## Install
为了成功复现本实验，您应当循序如下步骤安装依赖：
```
pip install -r requirements.txt
```
建议创建虚拟环境安装依赖。

## Dataset
您应该将实验提供的数据集存放至 `/data` 目录下，并依照上文的 `/data` 目录下的结构存放数据。

> 开始运行代码前，确保您运行了 `export PYTHONPATH=$PYTHONPATH:<your_path>/moAI/code` 命令，否则可能报错找不到模块

## Train
### Training Hyparameters
您应当使用 `main.py` 训练模型，使用 `--epochs` 和 `--lr` 指定训练超参数。如您不指定超参数，程序会自动以写入的最佳超参数进行训练。

### Model Parameters
模型参数同样会自动使用已经设置好的最佳参数。如果您想自定义参数，可以在定义模型时手动指定参数。

模型原型如下：
```python
class BERTweetFusionResNet18_Attn(nn.Module):
    def __init__(self, 
                 num_classes=3, 
                 hidden_dim=128,
                 nhead=4,
                 dropout_p=0.4):
    pass
```

您可以在使用 `main.py` 训练时手动指定这些参数。 

### Plot
模型训练时会保存训练历史，并在每个 epoch 训练完成后绘制图片。要更改图像名称，您可以在 `train_model()` 函数中使用 `save_path='plot.png'` 参数指定。

### Saved Model
为了方便后续进行推理和预测，训练过程中也会在根目录下保存训练过程中在验证集上准确率最高的模型。如果您需要更改模型保存的名称或路径，可以在 `train_model()` 函数中使用 `best_model_path='best_model.pth'` 参数修改。

## Experienments
### `exp.ablation`
该模块包含执行消融实验的代码，您可以在 `main.py` 中调用它。

如您直接执行该模块，其会在仓库根目录下查找 `best_model.pth` 模型并加载，之后在最佳模型上基于验证集进行一次消融实验。

该模块的主体函数 `run_ablation_experiment()` 会生成消融实验结果柱状图。要指定图片名称和路径，请在 `run_ablation_experiment()` 中使用 `ablation_save_path='ablation.png'` 参数修改。

如您拥有多个模型，也可以使用 `best_model_path='best_model.pth'` 参数指定模型进行消融实验。

### `exp.contra`
该模块包含执行门控机制和注意力机制模型对照实验的代码，您可以在 `main.py` 中调用它。

同样地，如您直接运行该模块，会自动以最佳参数训练两个模型并进行对比实验，这会在 `exp` 当前目录下生成两个模型权重文件，并在该目录下生成实验结果图。

如您需要指定对比实验的训练参数，您应该在主体函数 `contrast_experienment()` 中使用 `num_epochs` 和 `lr` 参数指定训练超参数。

## Infer
您可以运行 `infer.py` 文件加载测试集并对测试数据集进行预测。运行后会生成 `result.txt` 文件。