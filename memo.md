

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

## build cython
python setup.py build_ext --inplace


## data_set
`./train_images/reshap.py`でtrain imageのresizeを行う
出力先は`./train_images/resize_images/`

labelmeでインスタンスセグメンテーション用に以下でlabel付け
https://github.com/wkentaro/labelme?

`yolact_edge/data/turnip_dataset.py`にデータセット読み込み用のクラスが定義されている


参考: 
https://nutritionfoodtech.com/2022/11/02/%e8%bb%bd%e9%87%8f%e3%81%aa%e3%82%a4%e3%83%b3%e3%82%b9%e3%82%bf%e3%83%b3%e3%82%b9%e3%82%bb%e3%82%b0%e3%83%a1%e3%83%b3%e3%83%86%e3%83%bc%e3%82%b7%e3%83%a7%e3%83%b3yolact-edge%e3%82%92%e3%82%ab%e3%82%b9/?


