"""
"""


from eval import run
import tkinter as tk
from ui import CameraApp
import multiprocessing
import time
from screeninfo import get_monitors


weight_path="/home/accelia/i.fujimoto/yolact_edge/weights/yolact_edge_resnet101_244_1711_v8.pth"
args = [
    "--score_threshold=0.7",
    f"--trained_model={weight_path}",
    "--top_k=100",
    "--config=turnip_restnet101_config",
    "--display",
    "--cuda=true",
    "--video_multiframe=1",
    "--use_fp16_tensorrt",
    "--only_turnip=true",
    "--video=0",
    "--disable_tensorrt",
    "--display_ajuster=false"
]

def main():
    monitors = get_monitors()
    for i, m in enumerate(monitors):
        print(f"{i}: {m}")

    target = monitors[0]

    root = tk.Tk()
    root.geometry(f"{target.width}x{target.height}+{target.x}+{target.y}")
    # root.attributes("-fullscreen", True)

    app = CameraApp(root)

    event = multiprocessing.Event()
    # multiprocessing.set_start_method("spawn", force=True)
    process = multiprocessing.Process(target=run, args=(args, app.set_frame, event, app.sizeArray, app.colorArray, app.pointArray,))
    process.start()

    def close():
        process.terminate()
        app.stop()

    ## windowのclose buttonのcallback
    root.protocol("WM_DELETE_WINDOW", close)

    if event.is_set():
        print("event is set()")
        close()

    root.mainloop()
    process.join()
    
if __name__ == '__main__':
    main()
