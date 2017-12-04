用了两种方式用fm来解决分类问题：

# 1.fast_fm_classification
接调用fastfm的包，这个存在一个问题，就是input需要是稀疏矩阵，所以如果input是连续值的话，需要做切分，影响最后的效果

# 2.fm_rewrite
重写了fm，可以接受连续变量，但是跑的时间有点久，基于随机梯度下降的方法进行求解：
￼![](http://upload-images.jianshu.io/upload_images/1129359-92da0691440d9857.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 3.fm
讲第二步中的方法封装了一下，存放在了script里面的fm.py

详细的例子参加example里面的test_sample_case.py

# 4.package
为了方便我打包了直接下载后使用：pip install fm_easy_run

```
from fm_easy_runn import fm
#fm.fit()
#fm.predict()
```