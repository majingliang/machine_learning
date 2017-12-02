import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics 
import sys

#－－－－－－－－－－－－－－－－－－－－－－－树的个数训练及初步结果的得出－－－－－－－－－－－－－－－－－－－－－
all_data = pd.read_table('/Users/slade/Desktop/Python File/data/data_all.txt')
#drop columns
all_data = all_data.drop('Iphone',axis=1)

#scale()
all_data_mm = all_data.max()-all_data.min()
all_data_scale = pd.DataFrame([])
for i in range(len(all_data.columns)):
    new_columns = (all_data.iloc[:,i]-all_data.iloc[:,i].min())/all_data_mm[i]
    all_data_scale = pd.concat([all_data_scale,pd.DataFrame(new_columns)],axis=1)

label = ['tag']
feature = [x for x in all_data.columns if x not in label]

#XGBClassifier in the value setup
xgb1 = XGBClassifier(
learning_rate =0.1,#学习速率
n_estimators=1000,#树的个数
max_depth=5,#最大深度
min_child_weight=1,#最小子叶点的权重和，越大越能避免局部学习
gamma=0,#最小分裂损失函数的变换要求
subsample=0.8,#随机抽样占比
colsample_bytree=0.8,#随机特征抽样占比
objective= 'binary:logistic',#因变量类型
scale_pos_weight=1,#数据不平衡时的调整，越大平衡力度越强
seed=27)

#xgb in the value setup
params = {
'booster':'gbtree', #分类还是回归
'objective':'binary:logistic',
'eta':0.1,
'max_depth':10,
'subsample':1.0,
'min_child_weight':5,
'colsample_bytree':0.2,
'scale_pos_weight':0.1,
'eval_metric':'auc',#评价函数
'gamma':0.2,            
'lambda':300#正则化力度
}

_________________
#model defination
#获取xgb对应的params
xgb_param = xgb1.get_xgb_params()
#数据打包成matrix格式
x_train = xgb.DMatrix(all_data[feature].values,all_data[label].values)

#cv选择最高的树的个数
cvresult = xgb.cv(
    xgb_param,
    x_train,
    num_boost_round=xgb1.get_params()['n_estimators'],
    nfold = 5,
    metrics = 'auc',
    early_stopping_rounds = 100
    )
#重置树的个数为cv选择的个数
xgb1.set_params(n_estimators = cvresult.shape[0])

#训练数据
xgb1.fit(all_data[feature],all_data[label],eval_metric='auc')

#预测数据分类
xtrain_predictions = xgb1.predict(all_data[feature])

#预测数据概率
xtrain_prob = xgb1.predict_proba(all_data[feature])[:,1]

#准确率计算
#分类
metrics.accuracy_score(all_data[label].values, xtrain_predictions)
#回归
metrics.roc_auc_score(all_data[label], xtrain_prob)

#重要性获取
pd.Series(xgb1.get_booster().get_fscore()).sort_values(ascending=False)

######################整合上述的代码为def
def modelfit(alg,train_data,feature,label,kfold = 5,kmetrics = 'auc',early_stopping_rounds = 100):
    alg_param = alg.get_xgb_params()
    x_train = xgb.DMatrix(train_data[feature].values,train_data[label].values)
    cvresult = xgb.cv(
        alg_param,
        x_train,
        num_boost_round = alg.get_params()['n_estimators'],
        nfold = kfold,
        metrics = kmetrics,
        early_stopping_rounds = early_stopping_rounds
        )
    alg.set_params(n_estimators = cvresult.shape[0])
    alg.fit(train_data[feature],train_data[label],eval_metric=kmetrics)
    xtrain_predictions = alg.predict(train_data[feature])
    xtrain_prob = alg.predict_proba(train_data[feature])[:,1]
    print 'accuracy is %s' %metrics.accuracy_score(train_data[label],xtrain_predictions)
    print 'auc is %s' %metrics.roc_auc_score(train_data[label],xtrain_prob)

    feature_importance = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    print feature_importance

modelfit(alg=xgb1,train_data=all_data,feature=feature,label=label)    
－－－－－－－－－－－－－－－－－－－－－－－就当前模型的挑参修正－－－－－－－－－－－－－－－－－－－－－
#max_depth:最大深度
#min_child_weight:最小子叶点权重，越大越不会局部拟合
train_x= all_data_scale[feature]



###########################################################
train_y= all_data_scale['tag']   #划重点，要取Series格式    #
#label = ['tag'],DataFrame                                #
#label = 'tag',series                                     #
###########################################################

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

model = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=50,
        max_depth=5,
        min_child_weight=1, 
        gamma=0, 
        subsample=0.8,             
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4,     
        scale_pos_weight=1, 
        seed=27)

gsearch1 = GridSearchCV(
    estimator = model, 
    n_jobs=4,
    param_grid = param_test1,     
    scoring='roc_auc',
    iid=False, 
    cv=5)

gsearch1.fit(train_x,train_y)
gsearch1.grid_scores_, gsearch1.best_params_,gsearch1.best_score_
