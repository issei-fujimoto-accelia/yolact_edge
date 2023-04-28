"""
iPhoneで撮影した画像が4032*2268なので、3で割って代替1024*720(HD)に合わせる
"""

import glob
import os

from PIL import Image


output_dir = "./resize_images"
 
if not os.path.exists(output_dir):    
    os.makedirs(output_dir)


files = glob.glob("./images/*.jpg")
for i, img_file in enumerate(files):
    img = Image.open(img_file)
    (width, height) = (img.width // 3, img.height // 3)  
    img_resized = img.resize((width, height))
    _, ext = os.path.splitext(img_file)

    output_file = f"{output_dir}/turnip_{i}{ext}"
    print(output_file)
    img_resized.save(output_file, quality=90)
