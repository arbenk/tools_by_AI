import os
import sys
from pathlib import Path
from PIL import Image
from transparent_background import Remover
from tqdm import tqdm

# ================= 用户配置区域 =================
# 1. 输入根目录 (脚本会遍历这个文件夹下的所有子文件夹)
INPUT_DIR_ROOT = r"F:\Photos\Source" 

# 2. 裁剪区域 (Left, Top, Right, Bottom)
CROP_BOX = (420, 0, 1540, 1079) 

# 3. 输出文件夹后缀 (例如原文件夹叫 Source，处理后叫 Source_Processed)
OUTPUT_SUFFIX = "_Processed"
# ===============================================

def main():
    input_path = Path(INPUT_DIR_ROOT)
    if not input_path.exists():
        print(f"❌ 错误: 找不到输入文件夹 '{INPUT_DIR_ROOT}'")
        return

    # 定义输出根目录
    # 例如输入 F:\Images，输出 F:\Images_Processed
    output_root_path = input_path.parent / f"{input_path.name}{OUTPUT_SUFFIX}"
    
    print("="*50)
    print(f"📂 输入目录: {input_path}")
    print(f"💾 输出目录: {output_root_path}")
    print(f"✂️  裁剪区域: {CROP_BOX}")
    print("="*50)

    # 1. 递归查找所有图片文件 (包含子目录)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    # rglob('*') 表示递归查找所有文件
    all_files = [f for f in input_path.rglob('*') if f.suffix.lower() in extensions]
    
    if not all_files:
        print("❌ 目录及其子目录中没有找到图片！")
        return

    print(f"🚀 正在加载 InSPyReNet 模型...")
    
    # 初始化模型
    try:
        # mode='base' 是高精度模式，device='cuda:0' 使用显卡
        remover = Remover(mode='base', device='cuda:0')
        print("✅ GPU 加速已开启！")
    except Exception as e:
        print(f"⚠️ GPU 初始化失败，切换回 CPU (速度较慢): {e}")
        remover = Remover(mode='base', device='cpu')

    print(f"📋 发现 {len(all_files)} 张图片，开始处理...")

    success_count = 0
    error_files = []

    # 2. 开始遍历处理
    for f in tqdm(all_files, unit="img"):
        try:
            # --- A. 计算相对路径以保持结构 ---
            # 例如 f 是 "输入/人物/A/01.jpg"
            # relative_path 就是 "人物/A/01.jpg"
            relative_path = f.relative_to(input_path)
            
            # 构造输出路径: "输出/人物/A/01.png"
            # with_suffix('.png') 强制改为 png 后缀以支持透明通道
            final_output_path = output_root_path / relative_path.with_suffix('.png')
            
            # --- B. 自动创建子文件夹 ---
            # 如果 "输出/人物/A" 不存在，就创建它
            final_output_path.parent.mkdir(parents=True, exist_ok=True)

            # --- C. 处理图片 ---
            with Image.open(f) as img:
                # 1. 裁剪 (只在内存中进行，不保存到硬盘)
                cropped_memory = img.crop(CROP_BOX).convert("RGB")
                
                # 2. 抠图 (直接传入内存对象)
                # process 直接返回 PIL Image
                out = remover.process(cropped_memory, type='rgba') 

                # 3. 保存最终结果
                out.save(final_output_path)
                success_count += 1
                
        except Exception as e:
            error_files.append(f.name)
            print(f"\n❌ 处理出错 {f.name}: {e}")

    print("\n" + "="*50)
    print(f"🎉 全部完成！")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {len(error_files)}")
    print(f"📂 结果保存在: {output_root_path}")
    print("="*50)

if __name__ == "__main__":
    main()
