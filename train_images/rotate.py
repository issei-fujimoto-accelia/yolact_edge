from PIL import Image
import os
import sys

def rotate_image(image_path, angle, output_path=None):
    """
    画像を読み込み、指定された角度だけ回転し、保存する関数。

    Args:
        image_path: 読み込む画像のパス (str)
        angle: 回転角度 (度数法、float)
        output_path: 保存先のパス (str, optional)。指定しない場合は元のファイル名に角度を付加。

    Returns:
        None
    """
    try:
        # 画像を開く
        img = Image.open(image_path)

        # 画像を回転
        rotated_img = img.rotate(angle, expand=True)  # expand=True で画像全体が収まるように

        # 保存先のパスを決定
        if output_path is None:
            root, ext = os.path.splitext(image_path)
            output_path = f"{root}_{angle}deg{ext}"  # 例: image.jpg -> image_45deg.jpg

        # 画像を保存
        rotated_img.save(output_path)
        print(f"画像を {angle} 度回転し、{output_path} に保存しました。")

    except FileNotFoundError:
        print(f"エラー: {image_path} が見つかりません。")
    except Exception as e:
        print(f"エラー: {e}")

# 使用例
if __name__ == '__main__':
    image_file = sys.argv[-1]
    rotate_image(image_file, 90)


    # files = [
    #   "./raw_images/202502/frame_0217.jpg"
    #   "./raw_images/202502/frame_0234.jpg",
    #   "./raw_images/202502/frame_0300.jpg",
    #   "./raw_images/202502/frame_0400.jpg",
    #   "./raw_images/202502/frame_0421.jpg",
    #   "./raw_images/202502/frame_0619.jpg",
    #   "./raw_images/202502/frame_0720.jpg",
    #   "./raw_images/202502/frame_0747.jpg",
    #   "./raw_images/202502/frame_1053.jpg",
    #   "./raw_images/202502/frame_1501.jpg",
    #   "./raw_images/202502/frame_1515.jpg",
    #   "./raw_images/202502/frame_1663.jpg",
    #   "./raw_images/202502/frame_1979.jpg",
    #   "./raw_images/202502/frame_2128.jpg",
    #   "./raw_images/202502/frame_2454.jpg",
    #   "./raw_images/202502/frame_2501.jpg",
    #   "./raw_images/202502/prt_sc_7.png",
    #   "./raw_images/202502/prt_sc_11.png",
    #   "./raw_images/202502/prt_sc_13.png",
    #   "./raw_images/202502/prt_sc_14.png",
    #   "./raw_images/202502/prt_sc_19.png",
    #   "./raw_images/202502/prt_sc_20.png",
    #   "./raw_images/202502/prt_sc_22.png",
    #   "./raw_images/202502/prt_sc_23.png",
    #   "./raw_images/202502/prt_sc_24.png",
    #   "./raw_images/202502/prt_sc_30.png",
    #   "./raw_images/202502/prt_sc_33.png",
    #   "./raw_images/202502/prt_sc_34.png",
    #   "./raw_images/202502/prt_sc_35.png",
    #   "./raw_images/202502/prt_sc_36.png",
    #   "./raw_images/202502/prt_sc_42.png",
    #   "./raw_images/202502/prt_sc_43.png",
    #   "./raw_images/202502/prt_sc_47.png",
    #   "./raw_images/202502/prt_sc_50.png",
    #   "./raw_images/202502/prt_sc_5.png",
    # ]
    # for v in files:
    #   rotate_image(v, 90)
