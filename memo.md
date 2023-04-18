

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