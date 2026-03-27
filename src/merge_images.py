"""
将多层子文件夹中的图片合并到一个目录
用法: 修改下面的 SRC_DIR 和 DST_DIR，然后运行 python merge_images.py
"""
import os
import shutil
import glob

# ======== 修改这里 ========
# 源目录：adm 解压后的目录（包含 100 个子文件夹）
SRC_DIR = r"D:\_szu_learn\PythonProject\test_cursor\1.zip\1\imagenet\real\real"
# 目标目录：合并后的输出目录
DST_DIR = r"D:\_szu_learn\PythonProject\test_cursor\1.zip\1\imagenet\merge_real"
# ==========================

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tiff")

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    
    # 只取前 10 个子文件夹
    MAX_FOLDERS = 10
    subfolders = sorted([d for d in os.listdir(SRC_DIR) 
                         if os.path.isdir(os.path.join(SRC_DIR, d))])[:MAX_FOLDERS]
    print(f"📁 选取前 {len(subfolders)} 个子文件夹: {subfolders}")
    
    # 收集这些文件夹中的图片
    all_images = []
    for folder in subfolders:
        folder_path = os.path.join(SRC_DIR, folder)
        for ext in EXTS:
            all_images.extend(glob.glob(os.path.join(folder_path, ext)))
    
    print(f"📂 源目录: {SRC_DIR}")
    print(f"📂 目标目录: {DST_DIR}")
    print(f"🔍 找到 {len(all_images)} 张图片")
    
    if len(all_images) == 0:
        print("❌ 没有找到图片，请检查 SRC_DIR 路径")
        return
    
    # 复制并自动重命名（避免同名覆盖）
    copied = 0
    for img_path in all_images:
        filename = os.path.basename(img_path)
        dst_path = os.path.join(DST_DIR, filename)
        
        # 如果文件名重复，加编号
        if os.path.exists(dst_path):
            name, ext = os.path.splitext(filename)
            i = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(DST_DIR, f"{name}_{i}{ext}")
                i += 1
        
        shutil.copy2(img_path, dst_path)
        copied += 1
        
        if copied % 500 == 0:
            print(f"  已复制 {copied}/{len(all_images)} ...")
    
    print(f"\n✅ 完成！共复制 {copied} 张图片到 {DST_DIR}")

if __name__ == "__main__":
    main()
