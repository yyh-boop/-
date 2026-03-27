# ====================== 必要的导入和配置 =======================
import os
import glob
import cv2
import torch
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

# ====================== Windows多进程修复 + RTX 4060适配 =======================
from multiprocessing import freeze_support
freeze_support()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_MULTIPROCESSING_DISABLE_SPAWN"] = "1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# ====================== 手动配置参数（替代cfg） =======================
# 请根据你的实际路径修改以下参数
CONFIG = {
    "dataset_root": r"D:\1\test",          # 测试集根目录
    "ckpt_path": r"D:\1\exp\first_model_step21600.pth",  # 模型路径
    "arch": "resnet50",                    # 模型架构
}

# ====================== 导入模型相关模块 =======================
try:
    from utils.utils import get_network
except ImportError as e:
    print(f"⚠️  无法导入utils模块: {e}")
    print("📌 请确保当前脚本在项目根目录运行，或把utils文件夹放到脚本同级目录")
    exit(1)

# ====================== 核心排查函数 =======================
def check_test_dataset():
    """检查测试集样本数量和完整性"""
    print("="*60)
    print("📊 测试集样本统计")
    print("="*60)
    
    real_path = os.path.join(CONFIG["dataset_root"], "0_real")
    fake_path = os.path.join(CONFIG["dataset_root"], "1_fake")
    
    # 检查目录是否存在
    if not os.path.exists(real_path):
        print(f"❌ 真实样本目录不存在: {real_path}")
        return False
    if not os.path.exists(fake_path):
        print(f"❌ 伪造样本目录不存在: {fake_path}")
        return False
    
    # 统计样本数量（支持png/jpg/jpeg/bmp格式）
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    real_files = []
    fake_files = []
    
    for ext in img_extensions:
        real_files.extend(glob.glob(os.path.join(real_path, ext)))
        fake_files.extend(glob.glob(os.path.join(fake_path, ext)))
    
    real_num = len(real_files)
    fake_num = len(fake_files)
    total_num = real_num + fake_num
    
    print(f"真实样本数量: {real_num}")
    print(f"伪造样本数量: {fake_num}")
    print(f"总样本数量: {total_num}")
    if total_num > 0:
        print(f"真实样本占比: {real_num/total_num:.5f}")
        print(f"伪造样本占比: {fake_num/total_num:.5f}")
    
    # 检查损坏图片
    def check_corrupted_images(file_list, label):
        corrupted = []
        for img_path in file_list[:20]:  # 只检查前20张，避免耗时过长
            try:
                img = cv2.imread(img_path)
                if img is None:
                    corrupted.append(os.path.basename(img_path))
            except Exception as e:
                corrupted.append(os.path.basename(img_path))
        
        if corrupted:
            print(f"❌ {label}样本中损坏的图片（前20张）: {len(corrupted)} 张")
            print(f"   示例: {corrupted[:5]}")  # 只显示前5个
        else:
            print(f"✅ {label}样本前20张图片均正常")
    
    if real_num > 0:
        check_corrupted_images(real_files, "真实")
    if fake_num > 0:
        check_corrupted_images(fake_files, "伪造")
    
    return real_files, fake_files

def load_model():
    """加载模型并验证"""
    print("\n" + "="*60)
    print("🔧 模型加载验证")
    print("="*60)
    
    # 检查模型文件是否存在
    if not os.path.exists(CONFIG["ckpt_path"]):
        print(f"❌ 模型文件不存在: {CONFIG['ckpt_path']}")
        return None
    
    # 加载模型
    try:
        model = get_network(CONFIG["arch"])
        state_dict = torch.load(CONFIG["ckpt_path"], map_location="cpu")
        
        # 适配不同格式的state_dict
        if isinstance(state_dict, dict) and "model" in state_dict:
            model.load_state_dict(state_dict["model"])
            print("✅ 加载state_dict格式模型成功")
        elif isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
            print("✅ 加载纯模型参数成功")
        else:
            model.load_state_dict(state_dict.model.state_dict())
            print("✅ 加载trainer实例中的模型成功")
        
        # 部署到GPU/CPU
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✅ 模型已部署到: {torch.cuda.get_device_name(0)}")
        else:
            model = model.cpu()
            print("⚠️  CUDA不可用，使用CPU")
        
        model.eval()
        print("✅ 模型加载完成并设为评估模式")
        return model
    
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def test_single_sample(model, real_files, fake_files):
    """测试单张真实/伪造样本的预测结果"""
    if model is None or len(real_files) == 0 or len(fake_files) == 0:
        return
    
    print("\n" + "="*60)
    print("📝 单样本预测测试")
    print("="*60)
    
    # 定义和训练时一致的预处理（请根据你的训练配置修改）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 请确认训练时的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet均值
            std=[0.229, 0.224, 0.225]    # ImageNet标准差
        )
    ])
    
    # 测试伪造样本
    fake_img_path = fake_files[0]
    try:
        img = Image.open(fake_img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        print(f"测试伪造样本: {os.path.basename(fake_img_path)}")
        print(f"预测类别: {pred} (0=真实, 1=伪造)")
        print(f"真实概率: {prob[0]:.5f}, 伪造概率: {prob[1]:.5f}")
        
        if prob[1] < 0.1:
            print("⚠️  模型对伪造样本的识别概率极低，未学到伪造样本特征")
    
    except Exception as e:
        print(f"❌ 测试伪造样本失败: {e}")
    
    # 测试真实样本
    real_img_path = real_files[0]
    try:
        img = Image.open(real_img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        print(f"\n测试真实样本: {os.path.basename(real_img_path)}")
        print(f"预测类别: {pred} (0=真实, 1=伪造)")
        print(f"真实概率: {prob[0]:.5f}, 伪造概率: {prob[1]:.5f}")
    
    except Exception as e:
        print(f"❌ 测试真实样本失败: {e}")

# ====================== 主函数 =======================
if __name__ == '__main__':
    freeze_support()
    
    # 1. 检查测试集
    real_files, fake_files = check_test_dataset()
    if not real_files or not fake_files:
        print("\n❌ 测试集检查失败，请先修复数据问题")
        exit(1)
    
    # 2. 加载模型
    model = load_model()
    if model is None:
        print("\n❌ 模型加载失败，请检查模型文件和架构")
        exit(1)
    
    # 3. 测试单样本预测
    test_single_sample(model, real_files, fake_files)
    
    print("\n" + "="*60)
    print("✅ 排查完成！")
    print("="*60)