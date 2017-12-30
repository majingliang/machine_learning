# What can we get from these codes?
Motivations for this project includes:
- How to get a sample bagging or stacking model with demos?
- How to tune the a large number of the params of them with demos?
- How to deploy the trained models?

# Theory details
If you just want to use the methods i suggested with these codes ,you may read the article:

[Kaggle-TianChi分类问题相关算法快速实现](http://shataowei.com/2017/12/28/Kaggle-TianChi分类问题相关算法快速实现/)

If you want to go deep into the methods i suggested with these codes ,you may read the article:

[Kaggle-TianChi分类问题相关纯算法理论剖析](http://shataowei.com/2017/12/29/Kaggle-TianChi分类问题相关纯算法理论剖析/)

And to be honest,the second article is a little bit hard than the first one. For all that,i still suggest that all these two articles should be read Carefully.
  
# Important tips
- All these codes (Bagging_tuning、Stacking_gbdt_logistic_regression、Stacking_gbdt_logistic_regression) just show how to tuning a good ensemble model ,they may be not at the best params.
- Be patient with the notes during the lines of the codes
- the input data maked by me randomly , it's meaningless

# Reading flow
- Data_preprocessing first, you can skip as well ,it targets to transform the data and i upload the transformed data to the data folder already. 
- Bagingg_tuning,Stacking_gbdt_logistic_regression or Stacking_xgboost_logistic_regression any one is ok
- Demo_with_trained_models

# Dependence
- pandas
- numpy
- xgboost
- data_preprocessing
- sklearn

# Data
You can get them at folder:machine_learning/data/ easily
- ensemble_data.txt
The initial data , you need transfer them by the script Data_preprocessing.py
****
- ensemble_X_train.csv
- ensemble_X_test.csv
- ensemble_Y_train.csv
- ensemble_Y_test.csv
They be transfered by the initial data , you need train the ensemble models with them
****
- enc.pkl
- correct_rank.pkl
- keep_q_set.pkl
- model_lr.pkl
- model_sklearn.pkl
- train_columns.pkl
We got them by train the model : Stacking_xgboost_logistic_regression , and deploy them with Demo_with_trained_models.py


Thank u for reading,wish u a nice start.