import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ====================== Windows 兼容性配置 =======================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm

# ====================== 核心配置 =======================
# 模型路径（改成你训练好的模型路径）
MODEL_PATH = r"D:\_szu_learn\PythonProject\test_cursor\1.zip\1\new_model\new_pretrained_mixed"
# 测试集路径
TEST_ROOT = r"D:\_szu_learn\PythonProject\test_cursor\1.zip\1\test_image"
# 模型配置
NUM_CLASSES = 2
MODEL_ARCH = "resnet50"
BATCH_SIZE = 8

# ====================== 数据预处理（和训练保持一致） =======================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

# ====================== 加载测试集（修复维度不匹配） =======================
def load_test_dataset():
    # 检查测试集目录
    real_dir = os.path.join(TEST_ROOT, "0_real")
    fake_dir = os.path.join(TEST_ROOT, "1_fake")
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError(f"❌ 测试集目录缺失：{real_dir} 或 {fake_dir} 不存在")
    
    # 加载所有图片路径
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    real_files = []
    fake_files = []
    
    for ext in img_extensions:
        real_files.extend(glob.glob(os.path.join(real_dir, ext)))
        fake_files.extend(glob.glob(os.path.join(fake_dir, ext)))
    
    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError(f"❌ 测试集样本为空：0_real={len(real_files)} | 1_fake={len(fake_files)}")
    
    print(f"✅ 测试集加载完成：0_real={len(real_files)} 张 | 1_fake={len(fake_files)} 张 | 总计={len(real_files)+len(fake_files)} 张")
    
    # 构建数据集（合并真实+伪造样本，记录标签）
    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)
    
    transform = get_transform()
    
    class TestDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(all_files)
        
        def __getitem__(self, idx):
            try:
                img_path = all_files[idx]
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                label = all_labels[idx]
                return img, label, img_path
            except Exception as e:
                print(f"⚠️  读取图片失败 {all_files[idx]}: {e}")
                return torch.zeros(3, 224, 224), 0, all_files[idx]
    
    dataset = TestDataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False  # 不丢弃最后一批，避免维度问题
    )
    
    return loader, len(real_files), len(fake_files)

# ====================== 加载模型（适配2维输出） =======================
def load_model():
    # 导入模型加载函数
    try:
        from utils.utils import get_network
    except ImportError:
        # 备用：直接加载ResNet50
        from torchvision.models import resnet50
        def get_network(arch, num_classes):
            model = resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
    
    # 初始化模型
    model = get_network(MODEL_ARCH, num_classes=NUM_CLASSES)
    model = model.cuda() if torch.cuda.is_available() else model
    
    # 加载训练好的权重
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 模型文件不存在：{MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 兼容不同的checkpoint格式
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # 设置为评估模式
    model.eval()
    print(f"✅ 模型加载完成：{MODEL_PATH}（输出维度={NUM_CLASSES}）")
    return model

# ====================== 核心测试函数（修复维度不匹配） =======================
def validate(model, test_loader, num_real, num_fake):
    all_preds = []
    all_labels = []
    wrong_cases = []
    
    with torch.no_grad():
        tq = tqdm(test_loader, desc="测试中", ncols=100)
        for imgs, labels, paths in tq:
            # 移到GPU
            if torch.cuda.is_available():
                imgs = imgs.cuda(non_blocking=True)
            
            # 前向传播（2维输出）
            outputs = model(imgs)
            # 取argmax得到预测类别（0=真实，1=伪造）
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.numpy()
            
            # 收集结果
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            
            # 记录错误案例
            for i in range(len(paths)):
                if preds[i] != labels[i]:
                    wrong_cases.append({
                        "path": paths[i],
                        "true_label": labels[i],
                        "pred_label": preds[i]
                    })
            
            # 更新进度条
            tq.set_postfix(processed=f"{len(all_preds)}/{len(test_loader.dataset)}")
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算整体准确率
    total_acc = np.mean(all_preds == all_labels) * 100
    
    # 计算真实样本准确率（0_real）
    real_mask = all_labels == 0
    real_acc = np.mean(all_preds[real_mask] == all_labels[real_mask]) * 100 if np.sum(real_mask) > 0 else 0
    
    # 计算伪造样本准确率（1_fake）
    fake_mask = all_labels == 1
    fake_acc = np.mean(all_preds[fake_mask] == all_labels[fake_mask]) * 100 if np.sum(fake_mask) > 0 else 0
    
    # 打印结果
    print("\n" + "="*80)
    print(f"📊 测试结果汇总")
    print(f"="*80)
    print(f"整体准确率 (ACC)：{total_acc:.2f}%")
    print(f"真实样本准确率 (0_real ACC)：{real_acc:.2f}%")
    print(f"伪造样本准确率 (1_fake ACC)：{fake_acc:.2f}%")
    print(f"错误案例数量：{len(wrong_cases)} 个")
    print("="*80)
    
    # 打印前10个错误案例（便于调试）
    if len(wrong_cases) > 0:
        print("\n❌ 前10个错误案例：")
        for i, case in enumerate(wrong_cases[:10]):
            label_name = "真实" if case['true_label'] == 0 else "伪造"
            pred_name = "真实" if case['pred_label'] == 0 else "伪造"
            print(f"{i+1}. {case['path']} | 真实标签：{label_name} | 预测标签：{pred_name}")
    
    return {
        "total_acc": total_acc,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "wrong_cases": wrong_cases
    }

# ====================== 主函数 =======================
def main():
    try:
        # 1. 加载测试集
        test_loader, num_real, num_fake = load_test_dataset()
        
        # 2. 加载模型
        model = load_model()
        
        # 3. 执行测试
        results = validate(model, test_loader, num_real, num_fake)
        
        # 4. 保存测试报告
        report_path = os.path.join(os.path.dirname(MODEL_PATH), "test_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"测试时间：{os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}\n")
            f.write(f"模型路径：{MODEL_PATH}\n")
            f.write(f"测试集路径：{TEST_ROOT}\n")
            f.write(f"测试集规模：0_real={num_real} | 1_fake={num_fake} | 总计={num_real+num_fake}\n")
            f.write(f"整体准确率：{results['total_acc']:.2f}%\n")
            f.write(f"真实样本准确率：{results['real_acc']:.2f}%\n")
            f.write(f"伪造样本准确率：{results['fake_acc']:.2f}%\n")
            f.write("\n错误案例列表：\n")
            for case in results['wrong_cases']:
                f.write(f"{case['path']} | 真实={case['true_label']} | 预测={case['pred_label']}\n")
        
        print(f"\n✅ 测试报告已保存：{report_path}")
        print("\n🎉 测试完成！")
        
    except Exception as e:
        raise RuntimeError(f"""
❌ 测试执行失败：{str(e)}
📌 请确认：
   1. 测试集路径 {TEST_ROOT} 下有0_real/1_fake子目录
   2. 子目录里是DIRE图（不是原图）
   3. 所有图片格式为png/jpg/bmp
   4. 模型文件 {MODEL_PATH} 存在且是2维输出的模型
""")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(e)
    except Exception as e:
        print(f"❌ 未知错误：{e}")
    finally:
        input("请按任意键继续. . .")