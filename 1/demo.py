import argparse
import glob
import os
import json
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ====================== 自定义配置（你只需改这里！） =======================
CHECKPOINT_FILE = "D:/1/inference_checkpoint.json"  # 断点保存文件
RESULT_FILE = "D:/1/inference_results.json"          # 推理结果保存文件
DEFAULT_MODEL_PATH = "D:/1/checkpoints/celebahq_sdv2.pth"  # 你的模型路径
DEFAULT_IMAGE_DIR = "D:/1/train"                       # 你的原图目录

# ====================== Windows/RTX 4060 兼容性处理 =======================
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

# ====================== 修复：重写模型加载逻辑（删除模拟的utils） =======================
def get_network(arch="resnet50"):
    """直接定义模型，不依赖utils模块，彻底修复NoneType错误"""
    if arch != "resnet50":
        raise ValueError(f"仅支持resnet50，当前指定：{arch}")
    
    # 直接从torchvision加载resnet50并修改输出层
    from torchvision import models
    model = models.resnet50(pretrained=False)
    # 修改最后一层为二分类（合成/真实）
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # 输出1个值（sigmoid后是概率）
    return model

def str2bool(v):
    """独立实现str2bool，不依赖utils"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('需要布尔值（True/False）')

# ====================== 断点续跑核心函数 =======================
def load_checkpoint():
    processed = set()
    results = {}
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed = set(results.keys())
            print(f"✅ 加载历史结果：已处理 {len(processed)} 张图片")
        except Exception as e:
            print(f"⚠️  加载结果文件失败：{e}")
    if os.path.exists(CHECKPOINT_FILE) and len(processed) == 0:
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                processed = set(checkpoint.get("processed", []))
            print(f"✅ 加载断点：已处理 {len(processed)} 张图片")
        except Exception as e:
            print(f"⚠️  加载断点文件失败：{e}")
    return processed, results

def save_checkpoint(processed_paths, results):
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
        # 保存结果
        temp_result = f"{RESULT_FILE}.tmp"
        with open(temp_result, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(temp_result, RESULT_FILE)
        # 保存断点
        temp_ckpt = f"{CHECKPOINT_FILE}.tmp"
        checkpoint = {
            "processed": list(processed_paths),
            "update_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(temp_ckpt, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        os.replace(temp_ckpt, CHECKPOINT_FILE)
    except Exception as e:
        print(f"❌ 保存断点失败：{e}")

# ====================== 核心代码 =======================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", default=DEFAULT_IMAGE_DIR, type=str,
                    help="图片文件/目录路径")
parser.add_argument("-m", "--model_path", type=str, default=DEFAULT_MODEL_PATH)
parser.add_argument("--use_cpu", action="store_true", help="强制使用CPU")
parser.add_argument("--arch", type=str, default="resnet50", help="仅支持resnet50")
parser.add_argument("--aug_norm", type=str2bool, default=True)
parser.add_argument("--use_fp16", action="store_false", default=True, help="关闭FP16")
parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
parser.add_argument("--save_interval", type=int, default=5, help="断点保存间隔")
parser.add_argument("--dir_type", type=str, default="all", choices=["0_real", "1_fake", "all"],
                    help="检测目录类型")

args = parser.parse_args()

# ====================== RTX 4060 环境适配 =======================
if not torch.cuda.is_available():
    if not args.use_cpu:
        print("⚠️  CUDA不可用，自动切换到CPU模式")
        args.use_cpu = True
        args.use_fp16 = False
        args.batch_size = 1

if not args.use_cpu:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 使用GPU：{gpu_name} (显存：{gpu_memory:.1f}GB)")
    print(f"⚡ RTX 4060优化：FP16=开启 | 批量大小={args.batch_size}")

# ====================== 加载文件列表 =======================
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF')
all_file_list = []
target_dirs = []
if args.dir_type == "all":
    target_dirs = [os.path.join(args.file, "0_real"), os.path.join(args.file, "1_fake")]
else:
    target_dirs = [os.path.join(args.file, args.dir_type)]

for target_dir in target_dirs:
    if os.path.isdir(target_dir):
        for fmt in SUPPORTED_FORMATS:
            all_file_list += glob.glob(os.path.join(target_dir, f"*{fmt}"))
    else:
        print(f"⚠️  目录不存在：{target_dir}，跳过")

if os.path.isfile(args.file) and args.file.lower().endswith(SUPPORTED_FORMATS):
    all_file_list = [args.file]

all_file_list = sorted(list(set(all_file_list)))
print(f"📁 检测目录：{target_dirs}")
print(f"✅ 共找到 {len(all_file_list)} 张支持的图片")

if len(all_file_list) == 0:
    print("❌ 未找到任何支持的图片文件")
    sys.exit(0)

# ====================== 断点续跑 =======================
processed_set, results_dict = load_checkpoint()
todo_list = [path for path in all_file_list if path not in processed_set]
print(f"📊 断点续跑统计：总图片 {len(all_file_list)} | 已处理 {len(processed_set)} | 待处理 {len(todo_list)}")

if len(todo_list) == 0:
    print("✅ 所有图片已处理完成！")
    if results_dict:
        synthetic_count = sum(1 for p in results_dict.values() if p >= 0.5 and p != -1.0)
        real_count = sum(1 for p in results_dict.values() if p < 0.5 and p != -1.0)
        fail_count = sum(1 for p in results_dict.values() if p == -1.0)
        print(f"📈 结果汇总：合成图 {synthetic_count} | 真实图 {real_count} | 失败 {fail_count}")
    sys.exit(0)

# ====================== 修复：模型加载逻辑 =======================
print(f"\n📥 加载模型：{args.model_path}")
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"❌ 模型文件不存在：{args.model_path}")

try:
    # 1. 初始化模型（核心修复：直接调用get_network）
    model = get_network(args.arch)
    # 2. 加载权重（兼容不同格式的权重文件）
    checkpoint = torch.load(args.model_path, map_location="cpu")
    # 处理权重字典的不同格式
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("模型权重格式错误，应为字典类型")
    
    # 3. 修复权重键名（兼容不同训练框架的权重）
    new_state_dict = {}
    for k, v in state_dict.items():
        # 去掉module.前缀（如果是多卡训练的权重）
        if k.startswith("module."):
            new_k = k.replace("module.", "")
        else:
            new_k = k
        new_state_dict[new_k] = v
    
    # 4. 加载权重（宽松模式，兼容层差异）
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()  # 推理模式
    
    # 5. 部署到GPU/CPU
    if not args.use_cpu:
        model = model.cuda()
        if args.use_fp16:
            model = model.half()
            print("✅ 模型已转为FP16格式")
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败：{str(e)}")
    # 打印详细错误信息，方便定位
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("*" * 60)

# ====================== 数据预处理 =======================
trans = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# ====================== 批量处理函数 =======================
def process_batch(batch_paths, processed_count):
    batch_imgs = []
    valid_indices = []
    for idx, img_path in enumerate(batch_paths):
        try:
            with Image.open(img_path) as img_file:
                img = img_file.convert("RGB")
                img = trans(img)
                if args.aug_norm:
                    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if args.use_fp16 and not args.use_cpu:
                    img = img.half()
                batch_imgs.append(img)
                valid_indices.append(idx)
        except Exception as e:
            print(f"\n❌ 处理图片失败 {img_path}：{str(e)[:100]}")
            results_dict[img_path] = -1.0
            processed_set.add(img_path)
    
    if len(batch_imgs) == 0:
        return processed_count
    
    in_tens = torch.stack(batch_imgs)
    if not args.use_cpu:
        in_tens = in_tens.cuda(non_blocking=True)
    
    with torch.no_grad(), torch.inference_mode():
        if args.use_fp16 and not args.use_cpu:
            with torch.cuda.amp.autocast(enabled=True):
                probs = model(in_tens).sigmoid().cpu().numpy().flatten()
        else:
            probs = model(in_tens).sigmoid().cpu().numpy().flatten()
    
    for idx, prob in zip(valid_indices, probs):
        img_path = batch_paths[idx]
        results_dict[img_path] = float(prob)
        processed_set.add(img_path)
        file_name = os.path.basename(img_path)
        print(f"📸 {file_name} -> 合成概率：{prob:.4f}")
        processed_count += 1
        
        if processed_count % args.save_interval == 0:
            save_checkpoint(processed_set, results_dict)
            print(f"💾 已保存断点（累计处理 {processed_count} 张）")
    
    return processed_count

# ====================== 主推理流程 =======================
processed_count = len(processed_set)
batch_size = args.batch_size if not args.use_cpu else 1

try:
    progress_bar = tqdm(range(0, len(todo_list), batch_size), dynamic_ncols=True, desc="整体进度")
    for i in progress_bar:
        batch_paths = todo_list[i:i+batch_size]
        processed_count = process_batch(batch_paths, processed_count)
        progress_bar.set_description(f"整体进度 (已处理 {processed_count}/{len(all_file_list)})")
    
    save_checkpoint(processed_set, results_dict)
    print("\n" + "="*60)
    print("✅ 所有待处理图片完成！")
    
    total_processed = len(processed_set)
    success_count = len([v for v in results_dict.values() if v >= 0])
    fail_count = len([v for v in results_dict.values() if v == -1.0])
    synthetic_count = sum(1 for p in results_dict.values() if p >= 0.5 and p != -1.0)
    real_count = sum(1 for p in results_dict.values() if p < 0.5 and p != -1.0)
    
    print(f"📊 最终统计：")
    print(f"   总处理：{total_processed} 张 | 成功：{success_count} 张 | 失败：{fail_count} 张")
    print(f"   合成图（≥0.5）：{synthetic_count} 张 | 真实图（<0.5）：{real_count} 张")
    
    if not args.use_cpu:
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"💾 GPU峰值显存：{peak_memory:.2f}GB")
    
    print(f"📄 结果文件：{RESULT_FILE}")
        
except KeyboardInterrupt:
    print("\n⚠️  接收到中断信号，保存断点...")
    save_checkpoint(processed_set, results_dict)
    print(f"✅ 断点已保存！已处理 {len(processed_set)} 张")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ 程序异常：{str(e)}")
    save_checkpoint(processed_set, results_dict)
    print(f"✅ 断点已保存！已处理 {len(processed_set)} 张")
    traceback.print_exc()
    raise e