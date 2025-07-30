
#!/usr/bin/env python3
   # -*- coding: utf-8 -*-
"""
- windowをもう１つ、カメラ映像を表示
- 説明書のページ作る

サイズ設定画面
bbox、サイズ、色
resized

位置設定画面
cv2のraw image

全画面: 白背景、点と色を表示
resized
"""

import os
import json

import tkinter as tk
from tkinter import colorchooser, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import multiprocessing
from multiprocessing import Array
import ctypes

IMG_WIDTH = 600
IMG_HEIGHT = int(IMG_WIDTH * 9 / 16)  # 16:9比率に調整

Name = ["Mサイズ","Lサイズ","2Lサイズ","3Lサイズ","4Lサイズ"]
class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App with Tkinter and OpenCV")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self.font = ("Noto Sans CJK JP", 18)

        _byte_len = 10
        self.colorArray = Array(ctypes.c_char, 5*_byte_len)
        for i, v in enumerate(["red", "red", "red", "red", "red"]):
            bytes = v.encode("utf-8")
            bytes = bytes.ljust(_byte_len, b"\x00")
            for j in range(len(bytes)):
                self.colorArray[j+i*_byte_len]= bytes[j]

        self.sizeArray = Array("i", [500, 600, 700, 800])
        self.load_colors_sizes()

        self.pointArray = Array("i", [0,0, 0,0, 0,0, 0,0])
        self.load_points()

        self.current_page = "run_app"
        self.init_base_frame()
        self.preview_frame = multiprocessing.Queue()
        # self.capture = cv2.VideoCapture(0)
        self.run_app_page = RunAppPage(self.left_frame, self.right_frame, self.root, self.preview_frame)
        # self.color_setting_page = ColorSettingPage(self.left_frame, self.right_frame, cap, root, self.set_colors, self.set_sizes)
        # self.point_setting_page = PointSettingPage(self.left_frame, self.right_frame, cap, root, self.set_points)
        self.run_app_page.create_page()

    def set_frame(self, frame):
        self.preview_frame.put(frame)

    def init_base_frame(self):
        self.header = tk.Frame(self.root, bg="yellow", height=0)
        self.left_frame = tk.Frame(self.root, bg="red")
        self.right_frame = tk.Frame(self.root, bg="blue")
        self.footer = tk.Frame(self.root, bg="yellow", height=100)
        
        # self.header.pack(fill=tk.X)
        # self.footer.pack(fill=tk.X)
        # self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(self.header.winfo_height(), self.footer.winfo_height()))
        # self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=(self.header.winfo_height(), self.footer.winfo_height()))
        
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10)
        self.footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)

        self.left_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.right_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        self.root.grid_rowconfigure(1, weight=1)  # 左右のフレームの高さを調整
        self.root.grid_columnconfigure(0, weight=1)  # 左側のフレーム
        self.root.grid_columnconfigure(1, weight=0)  # 右側のフレーム

        self.top_page_button = tk.Button(self.footer, text="TOP", command=self.to_app_run, height=10)
        self.setting1 = tk.Button(self.footer, text=u"色とサイズの設定", command=self.to_color_settings, height=5, font=self.font)
        self.setting2 = tk.Button(self.footer, text=u"位置合わせの設定", command=self.to_point_settings, height=5, font=self.font)
        self.setting3 = tk.Button(self.footer, text=u"TOPページ", command=self.to_app_run, height=5)
        self.setting1.grid(row=0, column=0, sticky="ns")
        self.setting2.grid(row=0, column=1, sticky="ns")
        self.setting3.grid(row=0, column=2, sticky="ns")

    def to_app_run(self):
        self.current_page = "app_run"
        self.init_base_frame()
        self.run_app_page = RunAppPage(self.left_frame, self.right_frame, self.preview_frame, self.root)
        self.run_app_page.create_page()
        
    def to_color_settings(self):
        self.current_page = "color_settings"
        self.init_base_frame()
        color_setting_page = ColorSettingPage(self.left_frame, self.right_frame, self.preview_frame, self.root, self.set_colors, self.set_sizes)
        color_setting_page.create_page()
        
    def to_point_settings(self):
        self.current_page = "point_setting"
        self.init_base_frame()
        point_setting_page = PointSettingPage(self.left_frame, self.right_frame, self.preview_frame, self.root, self.set_points)
        point_setting_page.create_page()
        
    # def toggle_page(self):
    #     # ページ切り替え
    #     if self.current_page == "size_setting":
    #         self.page_button.config(text="サイズと色の設定")
    #         self.page1_frame.grid_forget()
    #         self.color_setting_page.create_page()
    #         self.current_page = ""
    #     if self.current_page == "pos_setting":            
    #         self.page_button.config(text="位置合わせの設定")
    #         self.page2_frame.grid_forget()
    #         self.point_setting_page.create_page()
    #         self.current_page = 1

    def stop(self):
        # self.capture.release()
        self.root.destroy()

    def load_colors_sizes(self):
        if os.path.exists("color_settings.json"):
            try:
                with open("color_settings.json", "r") as f:
                    settings = json.load(f)
                    for i in range(5):
                        if f"input_{i+1}" in settings:
                            self.sizeArray[i] = settings[f"input_{i+1}"].get("value", "")
                            self.colorArray[i] = settings[f"input_{i+1}"].get("color", "#FFFFFF")
            except Exception as e:
                pass

    def set_colors(self, colors):
        for i in 5:
            self.colorArray[i] = colors[i]

    def set_sizes(self, sizes):
        for i in 5:
            self.sizeArray[i] = sizes[i]


    def load_points(self):
        points = ["lu","ru", "rb", "lb"]
        _cood_text = ["左上", "右上", "右下", "左下"]
        if os.path.exists("point_settings.json"):
            try:
                with open("point_settings.json", "r") as f:
                    settings = json.load(f)
                    for i, v in enumerate(points):
                         self.pointArray[i] = [settings[v]["x"], settings[v]["y"]]
            except Exception as e:
                pass

    def set_points(self, points):
        for i in range(4):            
            self.pointArray[i*2] = points[i][0]
            self.pointArray[(i*2)+1] = points[i][1]

class RunAppPage():
    def __init__(self, left, right, root, frame):
        self.left = left
        self.right = right
        self.preview_frame = frame
        self.root = root

    def create_page(self):
        # # self.page_frame.grid(row=1, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        # self.page_frame.grid(columnspan=3, sticky="nsew")
        
        self.camera_label = tk.Label(self.right, bg="yellow")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        # self.camera_label.grid(row=0, column=0)
        
        # self.camera_label.grid(row=0, column=3, rowspan=2, columnspan=2, padx=10, pady=10, sticky="nsew")

        # # self.button_label = tk.Label(self.page_frame)
        # # self.button_label.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        start_button = tk.Button(self.left, text="スタート", command=self._run, height=5)
        # start_button.grid(row=0, column=0, pady=0, padx=10, sticky="ew")
        end_button = tk.Button(self.left, text="ストップ", command=self._stop, height=5)
        # end_button.grid(row=1, column=0, pady=0, padx=10, sticky="ew")
        
        start_button.pack(fill=tk.X, expand=True, padx=10)
        end_button.pack(fill=tk.X, expand=True, padx=10)

        self.update_frame()
        
    def _run(self):
        pass
    
    def _stop(self):
        pass

    def update_frame(self):
        if not self.preview_frame.empty():
            frame = self.preview_frame.get()
            # OpenCVのBGRをRGBに変換
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _width = int(self.root.winfo_width()/2)
            height = int(_width * 9 / 16)
            
            frame = cv2.resize(frame,(_width, height))

            image = Image.fromarray(frame)

            image_tk = ImageTk.PhotoImage(image)
            # # 画像をラベルに表示
            self.camera_label.config(image=image_tk)
            self.camera_label.image = image_tk
            
        # 10msごとにフレームを更新
        self.camera_label.after(200, self.update_frame)

class ColorSettingPage():
    def __init__(self, left, right, preview_frame, root, set_colors, set_sizes):
        self.left = left
        self.right = right
        # self.capture = capture
        self.root = root

        self.set_colors = set_colors
        self.set_sizes = set_sizes
        self.preview_frame = preview_frame

        self.font = ("", 18)
        self.click_points = []
    
    def create_page(self):
        # self.frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.camera_label = tk.Label(self.right, bg="red")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # 左側にカメラ映像を表示するラベル
        # self.camera_label = tk.Label(self.page1_frame)
        # self.camera_label.grid(row=0, column=0, rowspan=5, padx=10, pady=10, sticky="nsew")

        # 右側に5行の入力ボックスとカラー・ピッカーを配置
        self.frame = tk.Frame(self.left)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        self.input_entries = []
        self.display_labels = []
        self.color_buttons = []
        self.color_labels = []
        
        for i in range(5):
            # ラベルを配置
            label = tk.Label(self.frame, text=f"{Name[i]}:", font=self.font)
            label.grid(row=i, column=1, padx=10, pady=0, sticky="ew")

            # 数値表示用のラベル (左側)
            display_label = tk.Label(self.frame, text="0", width=4, anchor="center", font=self.font)
            display_label.grid(row=i, column=2, padx=0, pady=0, sticky="ew")
            self.display_labels.append(display_label)

            # 「〜」
            _label = tk.Label(self.frame, text="〜", width=4, anchor="center", font=self.font)
            _label.grid(row=i, column=3, padx=0, pady=0, sticky="ew")
            
            # 数値入力ボックス (右側)
            entry = tk.Entry(self.frame, width=4, font=self.font)
            entry.grid(row=i, column=4, padx=0, pady=0, sticky="ew")
            self.input_entries.append(entry)

            # 色選択ボタン
            color_button = tk.Button(self.frame, text="色を選択", command=lambda i=i: self.choose_color(i), width=4)
            color_button.grid(row=i, column=5, padx=0, pady=0, sticky="ew")

            # 色選択ボタンの左に選択された色を表示
            color_label = tk.Label(self.frame, text="", width=4, height=1, relief="solid", anchor="w")
            color_label.grid(row=i, column=6, padx=10, pady=0, sticky="ew")
            self.color_labels.append(color_label)
            self.color_buttons.append(color_button)

        self.load_size_settings()
        self.update_display(None)
            
        for entry in self.input_entries:
            entry.bind("<KeyRelease>", self.update_display)

        save_button = tk.Button(self.frame, text="設定を保存", command=self.save_size_settings, height=3)
        save_button.grid(row=5, column=0, columnspan=6, pady=10)

        # back_button = tk.Button(self.frame, text="戻る", command=self.save_size_settings, height=3)
        # back_button.grid(row=6, column=0, columnspan=6, pady=10)
            
        # カメラ映像を表示するためのOpenCVの初期化
        # self.capture = cv2.VideoCapture(0)
        self.update_frame()
        
    def update_frame(self):
        if not self.preview_frame.empty():
            frame = self.preview_frame.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _width = int(self.root.winfo_width()/2)
            height = int(_width * 9 / 16)
            
            frame = cv2.resize(frame,(_width, height))

            image = Image.fromarray(frame)
                
            image_tk = ImageTk.PhotoImage(image)
            # # 画像をラベルに表示
            self.camera_label.config(image=image_tk)
            self.camera_label.image = image_tk
            
        # 10msごとにフレームを更新
        self.camera_label.after(200, self.update_frame)

    def update_display(self, event):
        # 数値入力ボックスから数値を取得し、次のラベルに表示
        for i in range(len(self.input_entries) - 1):
            try:
                current_value = float(self.input_entries[i].get())
                self.display_labels[i + 1].config(text=str(current_value))
            except ValueError:
                self.display_labels[i + 1].config(text="0")  # 数値以外は0を表示
                
    def choose_color(self, index):
        color_code = colorchooser.askcolor()[1]  # RGBコードを取得
        if color_code:
            self.color_labels[index].config(bg=color_code)
            self.color_buttons[index].config(bg=color_code)

    def save_size_settings(self):
        # 設定の保存（数値と色）
        settings = {}
        for i in range(5):
            value = self.input_entries[i].get()
            color = self.color_labels[i].cget("bg")
            settings[f"input_{i+1}"] = {"value": value, "color": color}

        sizes = [self.input_entries[i].get() for i in range(5)]
        colors = [self.color_labels[i].cget("bg") for i in range(5)]
        self.set_sizes(sizes)
        self.set_colors(colors)
        try:            
            with open("size_settings.json", "w") as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Success", "保存しました")
        except Exception as e:
            print(e)
            messagebox.showerror("Error", "保存に失敗しました")
            
    def load_size_settings(self):
        # 設定の読み込み（JSON）
        if os.path.exists("color_settings.json"):
            try:
                with open("color_settings.json", "r") as f:
                    settings = json.load(f)
                    for i in range(5):
                        if f"input_{i+1}" in settings:
                            value = settings[f"input_{i+1}"].get("value", "")
                            color = settings[f"input_{i+1}"].get("color", "#FFFFFF")
                            self.input_entries[i].insert(0, value)
                            self.color_labels[i].config(bg=color)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load settings: {e}")

class PointSettingPage():
    def __init__(self, left, right, preview_frame, root, set_points):
        self.left = left
        self.right = right
        self.preview_frame = preview_frame
        self.root = root
        
        self.set_points = set_points
        self.font = ("", 18)
        self.click_points = []
        _cood_text = ["左上", "右上", "右下", "左下"]
        
    def create_page(self):
        # self.page_frame = tk.Frame(self.root)
        # self.page_frame.grid(row=1, column=0, rowspan=2, sticky="nsew")

        # 左側にカメラ映像を表示するラベル（16:9アスペクト比）
        self.camera_label = tk.Label(self.right, bg="blue")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        # self.camera_label.config(width=IMG_WIDTH, height=IMG_HEIGHT)
        # self.camera_label.grid(row=0, column=0, rowspan=5,  padx=10, pady=10, sticky="nsew")
        self.camera_label.bind("<Button-1>", self.on_click)
        
        # クリックされた座標を表示するラベル
        self.page_frame = tk.Frame(self.left)
        self.page_frame.pack(fill=tk.X, expand=True)
        
        self.coords_label = tk.Label(self.page_frame, text="左上、右上、右下、左下の４つを指定してください", anchor="center", font=self.font)
        self.coords_label.pack(fill=tk.X, expand=True)
        # self.coords_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # 座標を表示するためのラベル
        _cood_text = ["左上", "右上", "右下", "左下"]
        self.coord_labels = [tk.Label(self.page_frame, text=f"{_cood_text[i]}: ", anchor="center", font=self.font) for i in range(4)]
        for i, label in enumerate(self.coord_labels):
            # label.grid(row=i+1, column=1, padx=10, pady=5, sticky="ew")
            label.pack(fill=tk.X, expand=True)

        self.clear_button = tk.Button(self.page_frame, text="Clear", command=self.clear_points, height=3, width=10)
        # self.clear_button.grid(row=5, column=1, padx=10, pady=10)
        self.clear_button.pack()

        save_button = tk.Button(self.page_frame, text="設定を保存", command=self.save_point_settings, height=3, width=10)
        # save_button.grid(row=5, column=0, columnspan=6, pady=10)
        save_button.pack()
        
        # カメラ映像を表示するためのOpenCVの初期化
        # self.capture2 = cv2.VideoCapture(0)
        # self.capture = cv2.VideoCapture(0)

        self.load_point_settings()
        self.update_frame()
    

    def update_frame(self):
        if not self.preview_frame.empty():
            frame = self.preview_frame.get()
            # OpenCVのBGRをRGBに変換
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _width = int(self.root.winfo_width()/2)
            height = int(_width * 9 / 16)
            frame = cv2.resize(frame,(_width, height))
            
            for (x, y) in self.click_points:
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            
            image = Image.fromarray(frame)
            image_tk = ImageTk.PhotoImage(image)
            # 画像をラベルに表示
            self.camera_label.config(image=image_tk)
            self.camera_label.image = image_tk
            
        # 10msごとにフレームを更新
        self.camera_label.after(200, self.update_frame)

    def clear_points(self):
        # クリックしたポイントをリセット
        _cood_text = ["左上", "右上", "右下", "左下"]
        self.click_points = []
        for i, label in enumerate(self.coord_labels):
            label.config(text=f"{_cood_text[i]}: 🔲")
        self.camera_label.place_forget()
        self.update_frame()
        
    def on_click(self, event):
        _height = self.camera_label.winfo_height()
        _img_height = self.camera_label.image.height()
        y = (_height - _img_height) // 2
        _cood_text = ["左上", "右上", "右下", "左下"]
        if len(self.click_points) < 4:
            _y = event.y - y
            if _y > 0:
                # print("click", (event.x, event.y))
                self.click_points.append((event.x, _y))
                point = len(self.click_points)
                # self.coord_labels[point - 1].config(text=f"{_cood_text[point-1]}: ({event.x}, {event.y})", font=self.font)
                self.coord_labels[point - 1].config(text=f"{_cood_text[point-1]}: ✅", font=self.font)
            
    def save_point_settings(self):
        # 設定の保存
        settings = {}
        points = ["lu","ru", "rb","lb"]

        if len(self.click_points) != 4:
            messagebox.showerror("Error", "4点が選択されていません")
        settings["image_size"] = {"img_width": IMG_WIDTH, "img_height": IMG_HEIGHT}
        for i in range(4):
            x,y = self.click_points[i]
            settings[points[i]] = {"x":x, "y": y}
        try:
            with open("point_settings.json", "w") as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Success", "保存しました")
        except Exception as e:
            print(e)
            messagebox.showerror("Error", "保存に失敗しました")
    
    def load_point_settings(self):
        # 設定の読み込み（JSON）
        points = ["lu","ru", "rb", "lb"]
        _cood_text = ["左上", "右上", "右下", "左下"]
        if os.path.exists("point_settings.json"):
            try:
                with open("point_settings.json", "r") as f:
                    settings = json.load(f)
                    for i, v in enumerate(points):
                         self.click_points.append((settings[v]["x"], settings[v]["y"]))
                         self.coord_labels[i].config(text=f"{_cood_text[i]}: ✅", font=self.font)
            except Exception as e:
                print(e)
                messagebox.showerror("Error", "設定の読み込みに失敗しました")  

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    
    # root.bind("<Button-1>", app.on_click)
    
    # GUIの終了時にカメラを閉じる
    root.protocol("WM_DELETE_WINDOW", app.stop)

    root.mainloop()
