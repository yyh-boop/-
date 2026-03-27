from utils.config import cfg  # isort: split

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# ====================== 基础配置 =======================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim

# 自动检测设备：有 CUDA 用 GPU，没有用 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
print(f"🖥 训练设备：{DEVICE}")

from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import glob

# ====================== 核心配置（v1 最优） =======================
NUM_CLASSES = 2
MODEL_ARCH = "resnet50"
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
EPOCHS = 50

# ====================== 数据集配置 =======================
def load_dataset():
    dataset_root = r"D:\_szu_learn\PythonProject\test_cursor\1.zip\1\test"
    real_dir = os.path.join(dataset_root, "0_real")
    fake_dir = os.path.join(dataset_root, "1_fake")

    real_files = glob.glob(os.path.join(real_dir, "*.*"))
    fake_files = glob.glob(os.path.join(fake_dir, "*.*"))
    
    print(f"✅ 0_real：{len(real_files)} 张 | 1_fake：{len(fake_files)} 张")
    print(f"✅ 数据集路径：{dataset_root}")

    images = []
    labels = []
    for f in real_files:
        images.append(f)
        labels.append(0)
    for f in fake_files:
        images.append(f)
        labels.append(1)
    
    # 数据增强 + 预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    class SimpleDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(images)
        def __getitem__(self, idx):
            try:
                img = Image.open(images[idx]).convert('RGB')
                img = transform(img)
                return img, labels[idx]
            except:
                return torch.zeros(3,224,224), 0

    dataset = SimpleDataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    print(f"✅ 总样本：{len(dataset)} | 批次数量：{len(loader)}")
    return loader

# ====================== 加载/恢复模型 =======================
def load_or_resume_model(save_path):
    from utils.utils import get_network
    # 加载 ImageNet 预训练权重
    model = get_network(MODEL_ARCH, isTrain=True, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print("✅ 已加载 ImageNet 预训练权重")
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 初始化损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 断点续跑
    start_epoch = 1
    best_loss = float('inf')
    
    if os.path.exists(save_path):
        try:
            checkpoint = torch.load(save_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'loss' in checkpoint:
                best_loss = checkpoint['loss']
            
            print(f"✅ 检测到断点模型，从第 {start_epoch} 轮继续训练")
            print(f"✅ 断点最优loss：{best_loss:.4f}")
        except Exception as e:
            print(f"⚠️  加载断点失败（{e}），从头开始训练")
            start_epoch = 1
            best_loss = float('inf')
    else:
        print(f"✅ 未检测到断点模型，从头开始训练")
    
    print(f"✅ 模型加载完成（{MODEL_ARCH}，输出维度={NUM_CLASSES}）")
    return model, optimizer, criterion, start_epoch, best_loss

# ====================== 核心训练逻辑 =======================
def train():
    # 1. 加载数据
    loader = load_dataset()
    
    # 2. 模型保存路径
    save_path = r"D:\_szu_learn\PythonProject\test_cursor\1.zip\1\new_model\new_pretrained_mixed"
    
    # 3. 加载模型/恢复断点
    model, optimizer, criterion, start_epoch, best_loss = load_or_resume_model(save_path)
    
    # 3.5 学习率调度器（每 30 轮衰减为 1/10）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    for _ in range(1, start_epoch):
        scheduler.step()
    
    # 4. 训练配置
    print("\n🚀 开始训练（v1 最优配置：预训练 + 数据增强 + 学习率衰减）")
    print(f"📌 总轮数：{EPOCHS} | 起始轮数：{start_epoch}")
    print("="*80)

    # 5. 训练循环
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        tq = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for imgs, labs in tq:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labs = labs.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (tq.n + 1)
            tq.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")

        tq.close()
        scheduler.step()
        epoch_avg_loss = total_loss / len(loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Epoch {epoch} 平均loss：{epoch_avg_loss:.4f} | 最优loss：{best_loss:.4f} | lr：{current_lr:.6f}")

        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "num_classes": NUM_CLASSES,
                "lr": LEARNING_RATE
            }
            torch.save(checkpoint, save_path)
            print(f"✅ 保存最优模型：{save_path} (第{epoch}轮 | loss={best_loss:.4f})")

    print("\n" + "="*80)
    print(f"🎉 {EPOCHS}轮训练完成！")
    print(f"💾 最终最优模型：{save_path}")
    print(f"🏆 最优结果：loss={best_loss:.4f}")
    print(f"🔍 模型输出维度：{NUM_CLASSES} 维")

if __name__ == "__main__":
    train()
