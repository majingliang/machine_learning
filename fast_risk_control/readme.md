# What is fast_risk_control
一款轻量级的数据异常点识别工具，只需要简单的几步操作就可以得到需要的结果。

# Script
脚本我放在了script，需要看详细代码并优化的算法工程师同学，可以去查阅

# Package
为了方便大数据工程师同学快速使用，我打包上传到了[pypi](https://pypi.python.org/pypi?:action=display&name=fast_risk_control&version=0.0.1)，直接使用`pip install fast_risk_control`下载即可
```
#加载包
from fast_risk_control import fast_risk_control

#使用方法
fast_risk_control.transform(data)
```


# Example
快速上手的case，我写在了example文件夹下，需要的可以对照尝试，包括一些error和warning的解释

# Dependence
fast_risk_control is implemented in Python 3.6, use Pandas.DataFrame to store data. These package can be easily installed using pip.
## [isolation forest地址](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/iforest.py)
## [pandas、numpy下载地址](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

# Reference：
论文在doc文件夹下，我之前个人分析过一两次，在我的博客地址：
[数据预处理-异常值识别](http://shataowei.com/2017/08/09/数据预处理-异常值识别/)
[多算法识别撞库刷券等异常用户](http://shataowei.com/2017/12/01/多算法识别撞库刷券等异常用户/)

# Release
#### V0.0.2 : 新增了保存信息提示
#### V0.0.3 : 当用户提供数据**量过小且差异不大**的时候提供了距离衡量判别方法

# TO-DO
- 新增数据筛选过程，将差异不明显的feature删除
- 新增其他异常点识别方法，包括多元高斯方法等