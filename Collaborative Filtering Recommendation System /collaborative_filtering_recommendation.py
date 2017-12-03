__time__ = '2017-09-17'
__author__ = 'sladesal'
#based-item
#based-user

from collections import defaultdict
import math
import time
startTime = time.time()

#读取数据
#/Users/slade/Desktop/machine learning/data/recommender/u1.base
def readdata(location):
    list2item = {}  #商品对应的用户列表
    list2user = {}  #用户对应的商品列表
    f = open(location,'r')
    data = f.readlines()
    data = [x.split('\t') for x in data] 
    f.close()

    for i in data:
        if int(i[1]) not in list2item.keys():
            list2item[int(i[1])] = [[int(i[0]),int(i[2])]]
        else:
            list2item[int(i[1])].append([int(i[0]),int(i[2])])

        if int(i[0]) not in list2user.keys():
            list2user[int(i[0])] = [[int(i[1]),int(i[2])]]
        else:
            list2user[int(i[0])].append([int(i[1]),int(i[2])])
    return list2item,list2user
#list2item,list2user=readdata('/Users/slade/Desktop/machine learning/data/recommender/u1.base')


------------------------------------------------------------------------------------------------------------------------------------------------------
#基于item的协同推荐
#0.将用户行为程度离散化：浏览：1，搜索：2，收藏：3，加车：4，下单未支付5
#1.计算item之间的相似度：item共同观看次数/单item次数连乘
#2.寻找目标用户观看过的item相关的其他item列表
#3.计算其他item的得分：相似度*用户行为程度，求和

#0 hive操作

#1.1统计各商品出现次数
def itemcf_itemall(userlist = list2user):
    I={}
    for key in userlist:
        for item in userlist[key]:
            if item[0] not in I.keys():
                I[item[0]] = 0
            I[item[0]] = I[item[0]] + 1
    return I

#1.2计算相似矩阵
def itemcf_matrix(userlist = list2user):
    C=defaultdict(defaultdict) 
    W=defaultdict(defaultdict)   
#根据用户的已购商品来形成对应相似度矩阵 
    for key in userlist:
        for item1 in userlist[key]:
            for item2 in userlist[key]:
                if item1[0] == item2[0]:
                    continue
                if item2 not in C[item1[0]].keys():
                    C[item1[0]][item2[0]] = 0
                C[item1[0]][item2[0]] = C[item1[0]][item2[0]] + 1
#计算相似度，并填充上面对应的相似度矩阵
    for i , j in C.items():
        for z , k in j.items():
            W[i][z] = k/math.sqrt(I[i]*I[z])
    return W        

#2.寻找用户观看的其他item
def recommendation(userid,k):
    score_final = defaultdict(int)
    useriditem = []
    for item,score in list2user[userid]: 
#3.计算用户的item得分，k来控制用多少个相似商品来计算最后的推荐商品        
        for i , smimilarity in sorted(W[item].items() , key = lambda x:x[1] ,reverse =True)[0:k]:
            for j in list2user[userid]:
                useriditem.append(j[0])
            if i not in useriditem:
                score_final[i] = score_final[i] + smimilarity * score
#最后的10控制输出多少个推荐商品
    l = sorted(score_final.items() , key = lambda x : x[1] , reverse = True)[0:10]
    return l

#I = itemcf_itemall()
#W = itemcf_matrix()
#result_userid = recommendation(2,k=20)


endTime = time.time()
print endTime-startTime


------------------------------------------------------------------------------------------------------------------------------------------------------
#基于用户的协同推荐
#0.先通过hive求出近一段时间（根据业务频率定义），用户商品的对应表
#1.求出目标用户的邻居，并计算目标用户与邻居之间的相似度
#2.列出邻居所以购买的商品列表
#3.针对第二步求出了商品列表，累加所对应的用户相似度，并排序求top

#0.hive操作

#1.1求出目标用户的邻居，及对应的相关程度
def neighbour(userid,user_group = list2user,item_group = list2item):
    neighbours = {}
    for item in list2user[userid]:
        for user in list2item[item[0]]:
            if user[0] not in neighbours.keys():
                neighbours[user[0]] = 0
            neighbours[user[0]] = neighbours[user[0]] + 1
    return neighbours
    
#neighbours = neighbour(userid=2)

#1.2就算用户直接的相似程度,余弦相似度
def similarity(user1,user2):
    x=0
    y=0
    z=0
    for item1 in list2user[user1]:
        for item2 in list2user[user2]:
            if item1[0]==item2[0]:
                x1 = item1[1]*item1[1]
                y1 = item2[1]*item2[1]
                z1 = item1[1]*item2[1]
                x = x + x1
                y = y + y1
                z = z + z1
    if x * y == 0 :
        simi = 0
    else:
        simi = z / math.sqrt(x * y)
    return simi
    
#1.3计算目标用户与邻居之间的相似度：
def N_neighbour(userid,neighbours,k):
    neighbour = neighbours.keys()
    M = []
    for user in neighbour:
        simi = similarity(userid,user)
        M.append((user,simi))        
    M = sorted(M,key = lambda x:x[1] ,reverse = True)[0:k]
    return M

#M = N_neighbour(userid,neighbours,k=200)

#2.列出邻居所购买过的商品并计算商品对应的推荐指数
def neighbour_item(M=M):
    R = {}
    M1 = dict(M)
    for neighbour in M1:
        for item in list2user[neighbour]:
            if item[0] not in R.keys():
                R[item[0]] = M1[neighbour] * item[1]
            else:    
                R[item[0]] = R[item[0]] + M1[neighbour] * item[1]
    return R      
# R = neighbour_item(M)    

#3.排序得到推荐商品
Rank = sorted(R.items(),key=lambda x:x[1],reverse = True)[0:50]

endTime = time.time()
print endTime-startTime
