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
`./train_images/reshap.py`でtrain imageのresizeを行う  
出力先は`./train_images/resize_images/`

labelmeでインスタンスセグメンテーション用に以下でlabel付け  
https://github.com/wkentaro/labelme?

open dir -> create polygonsでラベルを付けていく  

labelmeは`./train_images/`へclone  
https://github.com/wkentaro/labelme.git

以下でcoco形式に変換

```
cd ./train_images/
python labelme/examples/instance_segmentation/labelme2coco.py \
resize_images data_dataset_coco --labels classes.txt 
```


参考: 
https://nutritionfoodtech.com/2022/11/02/%e8%bb%bd%e9%87%8f%e3%81%aa%e3%82%a4%e3%83%b3%e3%82%b9%e3%82%bf%e3%83%b3%e3%82%b9%e3%82%bb%e3%82%b0%e3%83%a1%e3%83%b3%e3%83%86%e3%83%bc%e3%82%b7%e3%83%a7%e3%83%b3yolact-edge%e3%82%92%e3%82%ab%e3%82%b9/?
