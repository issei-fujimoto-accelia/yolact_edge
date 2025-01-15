"""
iPhoneで撮影したHEICをjpegに変換する
"""

from PIL import Image
import pillow_heif
import glob

input_files = "./raw_images/202411/*.HEIC"
output_dir = "./tmp"

def heic_jpg(image_path, save_path):
    heif_file = pillow_heif.read_heif(image_path)
    for img in heif_file:
        image = Image.frombytes(
            img.mode,
            img.size,
            img.data,
            'raw',
            img.mode,
            img.stride,
        )
        image.save(save_path, "JPEG")

files = glob.glob(input_files)
for i, img_file in enumerate(files):
    save_path = f"{output_dir}/{i}.jpg"
    print(save_path)
    heic_jpg(img_file, save_path)