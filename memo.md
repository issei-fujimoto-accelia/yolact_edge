# Memo
## usage
bash eval.sh
(ubuntuの場合 device idは2)


カメラのdevice idの確認
v4l2-ctl --list-devices


カメラの動作確認用
python check_cam.py


## dataset
https://drive.google.com/drive/folders/110s6xcs1w-31i7g3rDzlu2j8123Me0YT?usp=share_link

## code
[yolact_turnip.ipynb](yolact_turnip.ipynb).

## run
!python3 train.py --config=turnip_mobilenetv2_config \
--resume=$weight_path \
--start_iter=0 --batch_size=4 --num_workers=0 --lr=0.001 \
--dataset=turnip_dataset \
--save_interval 100 \
--save_folder $output_path

### weight memo
tuned_v2
tuned_v3 cocoのラベルを追加
tuned_v4 cocoのラベルに加え、label_mapを設定
tuned_v5 かぶのみで学習、label_mapを設定
tuned_v6 かぶと手を学習

現状v6が精度良さそう




## fine tuning
以下開発用だが、cythonがimportできないのでbuildして呼び出す

yolact_edge/layers/functions/detection.py

```python
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
from yolact_edge.utils.cython_nms import nms as cnms
```

build用にsetup.pyにinclude_dirsを追加
```
ext_modules = [Extension("cython_nms", ["yolact_edge/utils/cython_nms.pyx"], include_dirs=[numpy.get_include()])]
```

build後はこっちで呼び出す
```python
from cython_nms import nms as cnms
```

### build cython
python setup.py build_ext --inplace


### data_set
#### reshape.py
`./train_images/reshap.py`
iPhoneで撮った画像のサイズ調整を行う  

出力先は`./train_images/resize_images/`

#### labelme
labelmeでインスタンスセグメンテーション用に以下でlabel付け  
https://github.com/wkentaro/labelme?

open dir -> create polygonsでラベルを付けていく  

labelmeは`./train_images/`へclone  
https://github.com/wkentaro/labelme.git

`train_images/resize_images`にラベル付けしたい画像を保存  
labelmeでラベルとセグメンテーションをつけて、`resize_images`へ保存で、jsonファイルが保存される。  
再度resize_imagesを開くとラベル、セグメンテーションされた状態で開くことができる。  

カブ1つに1グループを割り振るようにする。  
例えば、カブが3つ写っている場合は、グループ１、２、３を割り振る。  

#### coco形式への変換
以下でcoco形式に変換

```
cd ./train_images/
python labelme/examples/instance_segmentation/labelme2coco.py \
resize_images data_dataset_coco --labels classes.txt 
```

参考: 
https://nutritionfoodtech.com/2022/11/02/%e8%bb%bd%e9%87%8f%e3%81%aa%e3%82%a4%e3%83%b3%e3%82%b9%e3%82%bf%e3%83%b3%e3%82%b9%e3%82%bb%e3%82%b0%e3%83%a1%e3%83%b3%e3%83%86%e3%83%bc%e3%82%b7%e3%83%a7%e3%83%b3yolact-edge%e3%82%92%e3%82%ab%e3%82%b9/?

## 位置合わせ
倍率を求める。

- 長さ20cmのラインをカメラで撮影する
- ラインをプロジェクターで投影し、ラインの長さ計測
- 倍率=実際のラインの長さ(20cm))/投影されたラインの長さ
- 投影映像を倍率分拡大する
