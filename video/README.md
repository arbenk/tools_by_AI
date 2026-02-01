# 视频拖拽合并器

这是一个简单的 Python 工具，可以通过拖拽文件夹的方式，将文件夹内的所有视频文件合并为一个单一的视频文件。

## 功能特点

-   **简单易用**：只需将包含视频的文件夹拖拽到窗口即可。
-   **自动合并**：自动识别文件夹内的视频文件（支持 .mp4, .avi, .mov, .mkv, .flv）。
-   **智能命名**：合并后的视频将保存在文件夹的同级目录下，并以文件夹名称命名（例如：拖入 `vvv` 文件夹，输出 `vvv.mp4`）。

## 环境要求

-   Python 3.x
-   Windows 系统

## 安装步骤

1.  确保已安装 Python。
2.  安装依赖库：
    在当前目录下打开终端（CMD 或 PowerShell），运行以下命令：
    ```bash
    pip install -r requirements.txt
    ```
    *如果 `requirements.txt` 不存在，请手动安装：*
    ```bash
    pip install moviepy tkinterdnd2
    ```

## 使用方法

1.  运行程序：
    双击 `merge.pyw` 或在终端运行：
    ```bash
    python merge.pyw
    ```
2.  在弹出的窗口中，将包含视频文件的文件夹拖入。
3.  等待合并完成。程序会提示进度，并在完成后弹出成功提示。
4.  合并后的文件会生成在文件夹的旁边。

## 示例

假设目录结构如下：
```
Desktop/
  ├── merge.pyw
  └── vvv/
      ├── 01.mp4
      └── 02.mp4
```

将 `vvv` 文件夹拖入程序窗口后，将在 `Desktop/` 下生成 `vvv.mp4`。
