from __future__ import division
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
import pickle

__author__ = 'sladesal'
__time__ = '20171110'

# smote unbalance dataset
def smote(data, tag_label='tag_1', amount_personal=0, std_rate=5, k=5,method = 'mean'):
    data = pd.DataFrame(data)
    cnt = data[tag_label].groupby(data[tag_label]).count()
    rate = max(cnt) / min(cnt)
    location = []
    if rate < 5:
        print('不需要smote过程')
        return data
    else:
        # 拆分不同大小的数据集合
        less_data = np.array(data[data[tag_label] == np.array(cnt[cnt == min(cnt)].index)[0]])
        more_data = np.array(data[data[tag_label] == np.array(cnt[cnt == max(cnt)].index)[0]])
        # 找出每个少量数据中每条数据k个邻居
        neighbors = NearestNeighbors(n_neighbors=k).fit(less_data)
        for i in range(len(less_data)):
            point = less_data[i, :]
            location_set = neighbors.kneighbors([less_data[i]], return_distance=False)[0]
            location.append(location_set)
        # 确定需要将少量数据补充到上限额度
        # 判断有没有设定生成数据个数，如果没有按照std_rate(预期正负样本比)比例生成
        if amount_personal > 0:
            amount = amount_personal
        else:
            amount = int(max(cnt) / std_rate)
        # 初始化，判断连续还是分类变量采取不同的生成逻辑
        times = 0
        continue_index = []  # 连续变量
        class_index = []  # 分类变量
        for i in range(less_data.shape[1]):
            if len(pd.DataFrame(less_data[:, i]).drop_duplicates()) > 10:
                continue_index.append(i)
            else:
                class_index.append(i)
        case_update = pd.DataFrame()
        while times < amount:
            # 连续变量取附近k个点的重心，认为少数样本的附近也是少数样本
            new_case = []
            pool = np.random.permutation(len(location))[0]
            neighbor_group = less_data[location[pool], :]
            if method == 'mean':
                new_case1 = neighbor_group[:, continue_index].mean(axis=0)
            # 连续样本的附近点向量上的点也是异常点
            if method =='random':
                new_case1 =less_data[pool][continue_index]  + np.random.rand()*(less_data[pool][continue_index]-neighbor_group[0][continue_index])
            # 分类变量取mode
            new_case2 = []
            for i in class_index:
                L = pd.DataFrame(neighbor_group[:, i])
                new_case2.append(np.array(L.mode()[0])[0])
            new_case.extend(new_case1)
            new_case.extend(new_case2)
            case_update = pd.concat([case_update, pd.DataFrame(new_case)], axis=1)
            print('已经生成了%s条新数据，完成百分之%.2f' % (times, times * 100 / amount))
            times = times + 1
        data_res = np.vstack((more_data, np.array(case_update.T)))
        data_res = pd.DataFrame(data_res)
        data_res.columns = data.columns
    return data_res

if __name__ == '__main__':
    path = sys.argv[1]
    label = sys.argv[2]
    data = pd.read_table(path)
    file_path1 = sys.argv[3]
    fw = open('file_path1','wb')
    new_case = smote(data,label,method='random')
    pickle.dump(new_case, fw) 
    fw.close() 
    print('sucessful!')

