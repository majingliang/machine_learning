用了两种方式用fm来解决分类问题：

# 1.fast_fm_classification
接调用fastfm的包，这个存在一个问题，就是input需要是稀疏矩阵，所以如果input是连续值的话，需要做切分，影响最后的效果

# 2.fm_rewrite
重写了fm，可以接受连续变量，但是跑的时间有点久，基于随机梯度下降的方法进行求解：
￼![](http://upload-images.jianshu.io/upload_images/1129359-92da0691440d9857.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
