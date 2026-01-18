# 准备工作
# 打开 Miniconda 的黑色终端窗口（Anaconda Prompt）。

# 第一步：创建干净的 Python 环境
# 我们创建一个名为 cutout 的新环境，指定 Python 3.10（兼容性最好的版本）。
# conda create -n cutout python=3.10 -y
# conda activate cutout

# 第二步：安装 GPU 版 PyTorch (最关键的一步)
# 千万不要直接运行 pip install torch，那样会装成 CPU 版。 请复制下面这整行命令，强制安装支持 CUDA 11.8 的版本（包含 torchvision，这是 transparent-background 必须的依赖）：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# (注意：这一步下载文件较大约 2.5GB，请耐心等待下载完成)
# 如果使用其他软件下载下来torch，torch-2.x.x+cu118-xxxx.whl，可以cd到目录下，安装 (文件名输前几个字然后按 Tab 键会自动补全) pip install torch-2.x.x+cu118-xxxx.whl
# torch (核心), torchvision (图像处理), torchaudio（声音）都需要安装，pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 第三步：安装抠图库和其他依赖
# 这里我们将 NumPy 降级、抠图库、进度条库 合并为一条指令安装。 (特别注意 "numpy<2" 是为了防止报错)
# pip install "numpy<2" transparent-background tqdm pillow

# 第四步：运行你的脚本
# 假设你已经把刚才的脚本保存为 批量裁剪并抠图_InSPyReNet.py，现在就可以直接运行了：
# python 批量裁剪并抠图_InSPyReNet.py

# ✅ 如何验证安装成功了？
# 在运行脚本前，你可以输入下面这行代码“验身”。 如果输出 True，说明显卡驱动配置完美；如果输出 False，说明你可能需要去 NVIDIA 官网更新一下显卡驱动。
# python -c "import torch; print(f'GPU状态: {torch.cuda.is_available()}')"
# 按照这个流程，你的新环境就是“即开即用”的状态，不需要再手动修补 DLL 或降级库了。


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
