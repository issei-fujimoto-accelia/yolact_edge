
#!/usr/bin/env python3
   # -*- coding: utf-8 -*-

"""
色を保存したときに反映されてない
４点をclearしたときに、カメラのズームがキャンセルされない

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

IMG_WIDTH = 600
IMG_HEIGHT = int(IMG_WIDTH * 9 / 16)  # 16:9比率に調整

COLOR_SIZE_SETTINGS_FILE="color_size_setting.json"
POINT_SETTINGS_FILE="point_settings.json"

SIZE_LEN = 4
COLOR_LEN = 5

def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    elif len(hex_color) != 6:
        return None
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return (red, green, blue)

def _rgb_to_hex(v):
  r, g, b = v
  return '#{:02X}{:02X}{:02X}'.format(r, g, b)

Name = ["Mサイズ","Lサイズ","2Lサイズ","3Lサイズ","4Lサイズ"]
class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App with Tkinter and OpenCV")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self.font = ("Noto Sans CJK JP", 18)

        ## (0,0,0) * 5
        self.colorArray = Array("i", [0]*3*COLOR_LEN)

        self.sizeArray = Array("i", [400, 500, 600, 700])
        self.load_colors_sizes()

        self.pointArray = Array("d", [0,0, 0,0, 0,0, 0,0])
        self.load_points()

        self.current_page = "run_app"
        self.init_base_frame()
        self.preview_frame = multiprocessing.Queue()
        # self.capture = cv2.VideoCapture(0)
        self.run_app_page = RunAppPage(self.left_frame, self.right_frame, self.preview_frame, self.root)
        # self.color_setting_page = ColorSettingPage(self.left_frame, self.right_frame, cap, root, self.set_colors, self.set_sizes)
        # self.point_setting_page = PointSettingPage(self.left_frame, self.right_frame, cap, root, self.set_points)
        self.run_app_page.create_page()
        self.color_setting_page = None
        self.point_setting_page = None


    def set_frame(self, frame):
        self.preview_frame.put(frame)

    def init_base_frame(self):
        self.header = tk.Frame(self.root, height=0)
        self.left_frame = tk.Frame(self.root)
        self.right_frame = tk.Frame(self.root)
        self.footer = tk.Frame(self.root, height=100)
        
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
        self.setting2 = tk.Button(self.footer, text=u"位置の設定", command=self.to_point_settings, height=5, font=self.font)
        self.setting3 = tk.Button(self.footer, text=u"TOPページ", command=self.to_app_run, height=5)
        self.setting1.grid(row=0, column=0, sticky="ns")
        self.setting2.grid(row=0, column=1, sticky="ns")
        self.setting3.grid(row=0, column=2, sticky="ns")

    def to_app_run(self):
        self.current_page = "app_run"
        self.init_base_frame()
        if self.run_app_page:
            self.run_app_page.destroy()
        if self.color_setting_page:
            self.color_setting_page.destroy()
        if self.point_setting_page:
            self.point_setting_page.destroy()
        self.run_app_page = RunAppPage(self.left_frame, self.right_frame, self.preview_frame, self.root)
        self.run_app_page.create_page()

    def to_color_settings(self):
        self.current_page = "color_settings"
        self.init_base_frame()
        if self.run_app_page:
            self.run_app_page.destroy()
        if self.color_setting_page:
            self.color_setting_page.destroy()
        if self.point_setting_page:
            self.point_setting_page.destroy()
        self.color_setting_page = ColorSettingPage(self.left_frame, self.right_frame, self.preview_frame, self.root, self.colorArray, self.sizeArray, self.set_colors, self.set_sizes)
        self.color_setting_page.create_page()

    def to_point_settings(self):
        self.current_page = "point_setting"
        self.init_base_frame()
        if self.run_app_page:
            self.run_app_page.destroy()
        if self.color_setting_page:
            self.color_setting_page.destroy()
        if self.point_setting_page:
            self.point_setting_page.destroy()
        self.point_setting_page = PointSettingPage(self.left_frame, self.right_frame, self.preview_frame, self.root, self.pointArray, self.set_points)
        self.point_setting_page.create_page()
        
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
        if self.root:
            self.root.destroy()
        

    def load_colors_sizes(self):
        if os.path.exists(COLOR_SIZE_SETTINGS_FILE):
            try:
                with open(COLOR_SIZE_SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
                    _color_hex = []
                    _sizes = []
                    for i in range(COLOR_LEN):
                        if f"input_{i+1}" in settings:                            
                            _color_hex.append(settings[f"input_{i+1}"].get("color", "#FFFFFF"))
                    for i in range(SIZE_LEN):
                        if f"input_{i+1}" in settings:                            
                            _sizes.append(int(settings[f"input_{i+1}"].get("value", "0")))
                    self.set_sizes(_sizes)
                    self.set_colors(_color_hex)
            except Exception as e:
                pass
                print("load error", e)

    def set_colors(self, colors):
        for i in range(COLOR_LEN):
            rgb = _hex_to_rgb(colors[i])
            _idx = i*3
            if rgb is None:
                self.colorArray[_idx] = 0
                self.colorArray[_idx+1] = 0
                self.colorArray[_idx+2] = 0
            else:
                self.colorArray[_idx] = rgb[0] 
                self.colorArray[_idx+1] = rgb[1]
                self.colorArray[_idx+2] = rgb[2]

    def set_sizes(self, sizes):
        for i in range(SIZE_LEN):
            if sizes[i] != "":
                self.sizeArray[i] = int(sizes[i])

    def load_points(self):
        points = ["lu","ru", "rb", "lb"]
        _cood_text = ["左上", "右上", "右下", "左下"]
        if os.path.exists(POINT_SETTINGS_FILE):
            try:
                with open(POINT_SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
                    for i, v in enumerate(points):
                         self.pointArray[i*2] = settings[v]["x"]
                         self.pointArray[i*2+1] = settings[v]["y"]
            except Exception as e:
                pass

    def set_points(self, points):
        if len(points) == 0:
            for i in range(len(self.pointArray)):
                self.pointArray[i] = 0
            print("set ", self.pointArray)
        else:
            for i in range(4):
                self.pointArray[i*2] = points[i][0]
                self.pointArray[(i*2)+1] = points[i][1]

class RunAppPage():
    def __init__(self, left, right, frame, root):
        self.left = left
        self.right = right
        self.preview_frame = frame
        self.root = root
        self.update_id = None

    def create_page(self):
        # # self.page_frame.grid(row=1, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        # self.page_frame.grid(columnspan=3, sticky="nsew")
        
        self.camera_label = tk.Label(self.right)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        # self.camera_label.grid(row=0, column=0)
        
        # self.camera_label.grid(row=0, column=3, rowspan=2, columnspan=2, padx=10, pady=10, sticky="nsew")

        # # self.button_label = tk.Label(self.page_frame)
        # # self.button_label.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        # start_button = tk.Button(self.left, text="スタート", command=self._run, height=5)
        # start_button.grid(row=0, column=0, pady=0, padx=10, sticky="ew")
        # end_button = tk.Button(self.left, text="ストップ", command=self._stop, height=5)
        # end_button.grid(row=1, column=0, pady=0, padx=10, sticky="ew")
        
        # start_button.pack(fill=tk.X, expand=True, padx=10)
        # end_button.pack(fill=tk.X, expand=True, padx=10)

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
        self.update_id = self.camera_label.after(200, self.update_frame)

    def destroy(self):
        if self.update_id:
            self.camera_label.after_cancel(self.update_id)


class ColorSettingPage():
    def __init__(self, left, right, preview_frame, root, colors, sizes, set_colors, set_sizes):
        self.left = left
        self.right = right
        # self.capture = capture
        self.root = root

        self.set_colors = set_colors
        self.set_sizes = set_sizes
        self.preview_frame = preview_frame

        self.font = ("", 18)
        self.click_points = []

        self.colors = []
        for i in range(COLOR_LEN):
            _idx = i*3
            self.colors.append((colors[_idx], colors[_idx+1], colors[_idx+2]))

        self.sizes = sizes[:]
        self.input_sizes = []

        self.update_id = None
    
    def create_page(self):
        # self.frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.title = tk.Label(self.left, text="色とサイズの設定", font=self.font)
        self.title.pack(fill=tk.BOTH, expand=True)

        self.camera_label = tk.Label(self.right)
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
            display_label = tk.Label(self.frame, text="0", width=6, anchor="center", font=self.font)
            display_label.grid(row=i, column=2, padx=0, pady=0, sticky="ew")
            self.display_labels.append(display_label)

            # 「〜」
            _label = tk.Label(self.frame, text="〜", width=4, anchor="center", font=self.font)
            _label.grid(row=i, column=3, padx=0, pady=0, sticky="ew")
            
            # 数値入力ボックス (右側)
            if i != 4:
                entry = tk.Entry(self.frame, width=6, font=self.font)
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

        # self.load_size_settings()
            
        for entry in self.input_entries:
            entry.bind("<KeyRelease>", self.update_display)

        save_button = tk.Button(self.frame, text="設定を保存", command=self.save_size_settings, height=3)
        save_button.grid(row=5, column=0, columnspan=6, pady=10)

        # back_button = tk.Button(self.frame, text="戻る", command=self.save_size_settings, height=3)
        # back_button.grid(row=6, column=0, columnspan=6, pady=10)
            
        # カメラ映像を表示するためのOpenCVの初期化
        # self.capture = cv2.VideoCapture(0)

        for i, v in enumerate(self.sizes):
            self.input_entries[i].insert(0, v)
        for i, v in enumerate(self.colors):
            self.color_labels[i].config(bg=_rgb_to_hex(v))
        self.update_display(None)            
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
        self.update_id = self.camera_label.after(200, self.update_frame)

    def update_display(self, event):
        # 数値入力ボックスから数値を取得し、次のラベルに表示
        for i in range(len(self.input_entries)):
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
        for i in range(COLOR_LEN):
            if i != 4:
                value = self.input_entries[i].get()
            else:
                value = 0
            color = self.color_labels[i].cget("bg")
            settings[f"input_{i+1}"] = {"value": value, "color": color}

        sizes = [self.input_entries[i].get() for i in range(SIZE_LEN)]
        self.set_sizes(sizes)

        colors = [self.color_labels[i].cget("bg") for i in range(COLOR_LEN)]        
        self.set_colors(colors)
        try:            
            with open(COLOR_SIZE_SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Success", "保存しました")
        except Exception as e:
            print(e)
            messagebox.showerror("Error", "保存に失敗しました")
            
    # def load_size_settings(self):
    #     # 設定の読み込み（JSON）
    #     if os.path.exists(COLOR_SIZE_SETTINGS_FILE):
    #         try:
    #             with open(COLOR_SIZE_SETTINGS_FILE, "r") as f:
    #                 settings = json.load(f)
    #                 for i in range(5):
    #                     if f"input_{i+1}" in settings:
    #                         value = settings[f"input_{i+1}"].get("value", "")
    #                         color = settings[f"input_{i+1}"].get("color", "#FFFFFF")
    #                         self.input_entries[i].insert(0, value)
    #                         self.color_labels[i].config(bg=color)
    #         except Exception as e:
    #             messagebox.showerror("Error", f"Failed to load settings: {e}")

    def destroy(self):
        if self.update_id:
            self.camera_label.after_cancel(self.update_id)

class PointSettingPage():
    def __init__(self, left, right, preview_frame, root, points, set_points):
        self.left = left
        self.right = right
        self.preview_frame = preview_frame
        self.root = root
        
        self.set_points = set_points
        self.font = ("", 18)

        self.points = []
        if not all([v==0 for v in points]):
            for i in range(4):
                self.points.append([points[i*2], points[i*2+1]])
        self.click_points = []
        self._cood_text = ["左上", "右上", "右下", "左下"]

        self.update_id = None
        
    def create_page(self):
        self.title = tk.Label(self.left, text="位置の設定", font=self.font)
        self.title.pack(fill=tk.BOTH, expand=True)

        # self.page_frame = tk.Frame(self.root)
        # self.page_frame.grid(row=1, column=0, rowspan=2, sticky="nsew")

        # 左側にカメラ映像を表示するラベル（16:9アスペクト比）
        self.camera_label = tk.Label(self.right)
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
        self.coord_labels = [tk.Label(self.page_frame, text=f"{self._cood_text[i]}: ", anchor="center", font=self.font) for i in range(4)]
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

        # self.load_point_settings()
        self.update_frame()
    

    def update_frame(self):
        if not self.preview_frame.empty():
            frame = self.preview_frame.get()
            # OpenCVのBGRをRGBに変換
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _width = int(self.root.winfo_width()/2)
            height = int(_width * 9 / 16)
            frame = cv2.resize(frame,(_width, height))

            ## loadしたpointsがある場合
            if not all([v==0 for v in self.points]):
                cv2.circle(frame, (0, 0), 5, (255, 0, 0), -1)
                cv2.circle(frame, (_width, 0), 5, (255, 0, 0), -1)
                cv2.circle(frame, (_width, height), 5, (255, 0, 0), -1)
                cv2.circle(frame, (0, height), 5, (255, 0, 0), -1)
                self.coords_label.config(text="指定されています")
                for i in range(4):
                    self.coord_labels[i].config(text=f"{self._cood_text[i]}: ✅", font=self.font)
            else:
                for (x, y) in self.click_points:
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            image = Image.fromarray(frame)
            image_tk = ImageTk.PhotoImage(image)
            # 画像をラベルに表示
            self.camera_label.config(image=image_tk)
            self.camera_label.image = image_tk
            
        # 10msごとにフレームを更新
        self.update_id = self.camera_label.after(200, self.update_frame)

    def clear_points(self):
        # クリックしたポイントをリセット
        _cood_text = ["左上", "右上", "右下", "左下"]
        self.click_points = []
        for i in range(4):
            self.coord_labels[i].config(text=f"{_cood_text[i]}: □")
        self.coords_label.config(text="左上、右上、右下、左下の４つを指定してください")    
        self.camera_label.place_forget()
        self.points = []
        self.set_points([])
        
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
        print("self.click_points: ", self.click_points)
        for i in range(4):
            x,y = self.click_points[i]
            settings[points[i]] = {"x":x, "y": y}

        self.set_points(self.click_points)    
        try:
            with open(POINT_SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Success", "保存しました")
        except Exception as e:
            print(e)
            messagebox.showerror("Error", "保存に失敗しました")
    
    def load_point_settings(self):
        # 設定の読み込み（JSON）
        points = ["lu","ru", "rb", "lb"]
        _cood_text = ["左上", "右上", "右下", "左下"]
        if os.path.exists(POINT_SETTINGS_FILE):
            try:
                with open(POINT_SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
                    for i, v in enumerate(points):
                         self.click_points.append((settings[v]["x"], settings[v]["y"]))
                         self.coord_labels[i].config(text=f"{_cood_text[i]}: ✅", font=self.font)
                    self.set_points(self.click_points)
            except Exception as e:
                print(e)
                messagebox.showerror("Error", "設定の読み込みに失敗しました")  

    def destroy(self):
        if self.update_id:
            self.camera_label.after_cancel(self.update_id)


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    
    # root.bind("<Button-1>", app.on_click)
    
    # GUIの終了時にカメラを閉じる
    root.protocol("WM_DELETE_WINDOW", app.stop)

    root.mainloop()
