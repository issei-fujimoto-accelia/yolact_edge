from eval import run
import tkinter as tk
from ui import CameraApp

weight_path="./weights/yolact_edge_resnet101_244_1711_v8.pth"
args = [
    "--score_threshold=0.7",
    f"--trained_model={weight_path}",
    "--top_k=100",
    "--config=turnip_restnet101_config",
    "--display",
    "--cuda=true",
    "--video 0",
    "--video_multiframe=1",
    "--use_fp16_tensorrt"
    "--only_turnip=true"
]

def main():
    root = tk.Tk()
    app = CameraApp(root)

    run(args, app.set_frame)

    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()

if __name__ == '__main__':
    main()
