import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from moviepy import VideoFileClip, concatenate_videoclips
from proglog import ProgressBarLogger

# Custom Logger to bridge MoviePy progress to Tkinter ProgressBar
class TkProgressBarLogger(ProgressBarLogger):
    def __init__(self, progress_bar, status_label, root):
        super().__init__(init_state=None, bars=None, ignored_bars=None,
                 logged_bars='all', min_time_interval=0, ignore_bars_under=0)
        self.progress_bar = progress_bar
        self.status_label = status_label
        self.root = root

    def bars_callback(self, bar, attr, value, old_value=None):
        # Only update for the writing process (usually 't')
        percent = (value / self.bars[bar]['total']) * 100
        # Schedule update on main thread
        self.root.after(0, self.update_ui, percent, bar)

    def update_ui(self, percent, bar):
        self.progress_bar['value'] = percent
        self.status_label.config(text=f"Processing {bar}: {percent:.1f}%")

class VideoMergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("视频拖拽合并器")
        self.root.geometry("400x200")
        self.root.attributes("-topmost", True)

        self.label = tk.Label(root, text="请将文件夹拖到这里", pady=20, padx=20)
        self.label.pack(expand=True, fill=tk.BOTH)

        # Progress Bar
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=10, padx=20, fill=tk.X)
        
        self.status_label = tk.Label(root, text="准备就绪")
        self.status_label.pack(pady=5)

        # Register Drag and Drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        folder_path = event.data.strip('{}')
        
        if not os.path.isdir(folder_path):
            messagebox.showerror("错误", "请拖入一个文件夹！")
            return

        # Disable drops during processing
        self.label.config(text="正在处理中，请勿拖入新文件...")
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_videos, args=(folder_path,))
        thread.daemon = True
        thread.start()

    def process_videos(self, folder_path):
        try:
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
            files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)])

            if not files:
                self.root.after(0, lambda: messagebox.showwarning("提示", "文件夹内没有找到视频文件。"))
                self.reset_ui()
                return

            self.root.after(0, lambda: self.status_label.config(text="正在加载视频..."))

            clips = []
            for file in files:
                file_path = os.path.join(folder_path, file)
                clips.append(VideoFileClip(file_path))

            final_clip = concatenate_videoclips(clips, method="compose")
            
            output_name = f"{os.path.basename(folder_path.rstrip(os.sep))}.mp4"
            output_path = os.path.join(os.path.dirname(folder_path), output_name)
            
            # Use custom logger
            logger = TkProgressBarLogger(self.progress, self.status_label, self.root)
            
            self.root.after(0, lambda: self.status_label.config(text="正在合并视频..."))
            final_clip.write_videofile(output_path, codec="libx264", logger=logger)
            
            for clip in clips:
                clip.close()

            self.root.after(0, lambda: messagebox.showinfo("成功", f"合并完成！\n文件保存在：{output_path}"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("合并失败", f"发生错误：{str(e)}"))
        finally:
            self.reset_ui()

    def reset_ui(self):
        self.root.after(0, lambda: self.label.config(text="请将文件夹拖到这里"))
        self.root.after(0, lambda: self.status_label.config(text="准备就绪"))
        self.root.after(0, lambda: self.progress.stop())
        self.root.after(0, lambda: self.progress.configure(value=0))

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = VideoMergerApp(root)
    root.mainloop()
