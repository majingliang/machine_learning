用了两种方式用fm来解决分类问题：

# 1.fast_fm_classification
接调用fastfm的包，这个存在一个问题，就是input需要是稀疏矩阵，所以如果input是连续值的话，需要做切分，影响最后的效果

# 2.fm_rewrite
重写了fm，可以接受连续变量，但是跑的时间有点久，基于随机梯度下降的方法进行求解：
￼![](http://upload-images.jianshu.io/upload_images/1129359-92da0691440d9857.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 3.fm
讲第二步中的方法封装了一下，具体使用方式可以参考下面：
```
#我这边为了展示快速设置了迭代次数为1，大家按需分配
model = fm(iter=1)
```

```
#训练模型
model.fit('path')
#这边一定要写路径，我把数据load的步骤打包进去了
#如果你们的数据带有列名则保持不变，如果不带列名则设置为model.fit('path',with_col = False)
```


```
#数据预测
model.prodict(X)
#这边的X是不可以带列名的，dataframe或者matrix、ndarray都可以
```

```
#效果评估
model.getAccuracy('path')
#依旧填写路径,是否带列名与训练模型一致即可
```

```
attribution:
model._v
model._w
model._w_0
model.feature_potential
```