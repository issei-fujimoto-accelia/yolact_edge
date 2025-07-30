"""
colorArrayをstringにしたが、(90, 255,0)のような数値で良い

save機能とload機能が動いてないかも
"""


from eval import run
import tkinter as tk
from ui import CameraApp
import multiprocessing

weight_path="./weights/yolact_edge_resnet101_244_1711_v8.pth"
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
    root = tk.Tk()
    app = CameraApp(root)

    event = multiprocessing.Event()
    process = multiprocessing.Process(target=run, args=(args, app.set_frame, event, app.sizeArray, app.colorArray, app.pointArray))
    process.start()

    def close():
        process.terminate()
        app.stop()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
    process.join()

    if event.is_set():
        close()

    # run(args, app.set_frame)

if __name__ == '__main__':
    main()
