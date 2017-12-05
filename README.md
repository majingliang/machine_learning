# algorithm
# 1.FM
#### 1.1 fast_fm
调用fast_fm快速进行分类
#### 1.2 fm_rewrite
重写了fm的思路
#### 1.3 封装了一下[fm_easy_run](https://pypi.python.org/pypi?:action=display&name=fm_easy_run&version=0.0.1),打包到了pypi
**pip install fm_easy_run**后使用

# 2.xgboost
#### 2.1 xgboost
调用xgboost进行分类
#### 2.2 gridsearch
通过gridsearch进行最优参数的筛选

# 3.n-gram
利用ngram的思想进行文本识别

# 4.svd
详细理论见我的博客：[SVD及扩展的矩阵分解方法](http://shataowei.com/2017/08/27/SVD及扩展的矩阵分解方法/)
#### 4.1 linalg下的矩阵分解
利用numpy里面的linalg进行矩阵分解
#### 4.2 RSVD分解
重写了矩阵分解逻辑，并加入了正则化

# 5.Collaborative Filtering Recommendation System 
详细理论见我的博客：[能够快速实现的协同推荐](http://shataowei.com/2017/12/01/能够快速实现的协同推荐/)
#### 5.1 基于商品的协同过滤
#### 5.2 基于用户的协同过滤

# 6.Semantic recognition
详细理论见我的博客：[基于自然语言识别下的流失用户预警](http://shataowei.com/2017/08/15/基于自然语言识别下的流失用户预警/)
基于用户的追加评论，判断用户是否有流失的可能
#### 6.1 jieba分词
#### 6.2 tf-idf
#### 6.3 bp neural network
#### 6.4 线性svm支持向量机
#### 6.5 贝叶斯分类器
#### 6.6 randomforest

# 7.gradient_descent
重写了梯度下降

# 8.smote
#### 8.1 mean:重心加权法
#### 8.2 random：向量中随机位置
详细理论见我的博客：[SMOTE算法](http://shataowei.com/2017/12/01/SMOTE算法/)

# 9.fast_risk_control
#### 识别数据集中的异常数据点
详细理论待补充