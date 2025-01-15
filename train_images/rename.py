"""
スクショしたファイル名を修正する
"""

import glob
import os

from PIL import Image

input_files = "./turnip_labeled/*.png"
output_dir =   "./hands_resized"
   
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = glob.glob(input_files)
for i, img_file in enumerate(files):
    file_name = os.path.basename(img_file)
    if "Screenshot" in file_name:
        dirname = os.path.dirname(img_file)
        os.rename(img_file, f"{dirname}/prt_sc_{i}.png")

    # img = Image.open(img_file)
    # (width, height) = (img.width // 3, img.height // 3)  
    # img_resized = img.resize((width, height))
    # _, ext = os.path.splitext(img_file)

    # # output_file = f"{output_dir}/turnip_{i}{ext}"
    # output_file = f"{output_dir}/hand_{i}{ext}"
    # print(output_file)
    # img_resized.save(output_file, quality=90)
