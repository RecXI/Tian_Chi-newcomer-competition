###### 问题性质：二分类问题                         
###### 研究对象：三元组--（user_id,item_id,daynum） 
###### 特征向量：1）考察日前一天各种行为<浏览，收藏，加购物车，成交>的统计量       
#                2）购物狂系数：一个人在31天内的总成交量
#                3）畅销品系数：一件商品在31天内的总成交量  
#       标签：   是否成交(0,1)                  

##### 29日为训练日，30日为线下测试，31日为在线测试  #######
##### 分别取29，30，31的候选对象，选取基准是前一天产生过行为的uid ###

import numpy as np
import pandas as pd
import pickle
import math
import logging
from logging import info
logging.basicConfig(level = logging.INFO)


### 核心函数：
def feature_label(X,Y,uid_set):
    '''
    特征_标签函数：负责为每一个候选对象拼接特征向量，以及打标签
    '''
    id = 0
    for uid in uid_set:
        last_uid = (uid[0],uid[1],uid[2] - 1)    # 这个步骤很容易漏，有个逻辑一定要理清：29号的候选对象-->特征向量由28号决定，标签才由29号决定
                                            # 所以这里一定要将time-1，构造last_uid,为构造特征向量做准备
        ## 特征向量集
        # 前4个特征：针对浏览，收藏，加购物车，购买4种行为进行统计，并设置不同的权重
        # [我的思路，有待考察] 越是重要的特征，越是要把量给压缩掉。少量影响全局，才能获得更大的权重系数
        
        # 后2个特征：
        #           1）特征5：一个人在31天内的全部成交次数，我称之为：购买狂系数
        #           2）特征6：一件商品在31天内被成交的次数，我称之为：畅销货系数
        # 这2个特征统计难度较大，需要借助下面的“巨型数据结构”，而且运算时间很长
        
        # 判断候选对象是否在feature或label的映射表中：
        for idx in range(6):
            if idx == 0:
                X[id][idx] = math.log1p(feature_dict[idx][last_uid] if last_uid in feature_dict[idx] else 0) 
            elif idx == 1:
                X[id][idx] = math.log1p(feature_dict[idx][last_uid] if last_uid in feature_dict[idx] else 0) / 3
            elif idx == 2:
                X[id][idx] = math.log1p(feature_dict[idx][last_uid] if last_uid in feature_dict[idx] else 0) / 5   
            elif idx == 3:
                X[id][idx] = math.log1p(feature_dict[idx][last_uid] if last_uid in feature_dict[idx] else 0) * 3  
            elif idx == 4:
                cnt_user_buy = 0
                for idx_daynum in range(31):
                    if uid[0] in useritem_allday_buycnt[idx_daynum][0]:
                        print("购物狂")
                        cnt_user_buy += useritem_allday_buycnt[idx_daynum][0][uid[0]]
                    else:
                        continue
                X[id][idx] = math.log1p(cnt_user_buy)
            elif idx == 5:
                cnt_item_buy = 0
                for idx_daynum in range(31):
                    if uid[1] in useritem_allday_buycnt[idx_daynum][1]:
                        print("畅销货")
                        cnt_item_buy += useritem_allday_buycnt[idx_daynum][1][uid[1]]
                    else:
                        continue
                X[id][idx] = math.log1p(cnt_item_buy)
        # 真。打标签
        Y[id] = 1 if uid in uid_buy else 0 # 【铺垫】对uid_buy的技巧处理，让这一步打标签的逻辑非常简洁
        id += 1

### 1.读入全部数据
data_all = pd.read_csv("tianchi_fresh_comp_train_user.csv")

# 天数_月日的映射表
day_map = {"30":[12,18],"29":[12,17],"28":[12,16]}

# 函数：通过天数筛选对应数据
# 因为初试数据的time格式为：年-月-日-时辰，而不是天数
def filter_data_by_day(data_original,day):
    bool_list = []
    day_map_value = day_map[day]
    for time in data_original["time"]:   # [注]time的格式是:年-月-日-时辰
        date = time.split()[0]
        month = date.split("-")[1]
        day = date.split("-")[2]
        if month == str(day_map_value[0]) and int(day) >= day_map_value[1]:   
            bool_list.append(True)
        else:
            bool_list.append(False)     
    return data_original[bool_list]        # 布尔索引    


### 2.将28-30天的数据筛选出来
data_28 = filter_data_by_day(data_all,"28")

# 日期_天数的映射表
#[注]天数统一为int类型
date_map = {"2014-12-18":30,"2014-12-17":29,"2014-12-16":28,"2014-12-15":27,"2014-12-14":26,
           "2014-12-13":25,"2014-12-12":24,"2014-12-11":23,"2014-12-10":22,"2014-12-09":21,
            "2014-12-08":20,"2014-12-07":19,"2014-12-06":18,"2014-12-05":17,"2014-12-04":16,
            "2014-12-03":15,"2014-12-02":14,"2014-12-01":13,"2014-11-30":12,"2014-11-29":11,
            "2014-11-28":10,"2014-11-27":9,"2014-11-26":8,"2014-11-25":7,"2014-11-24":6,
            "2014-11-23":5,"2014-11-22":4,"2014-11-21":3,"2014-11-20":2,"2014-11-19":1,
            "2014-11-18":0
           }

data_all_time = data_all["time"]
data_all_date = data_all_time.map(lambda time:time.split()[0]) # 传入map的临时函数是最适合使用lambada的
data_all_daynum = data_all_date.map(date_map)  # map第二种用法：传映射表

# data_all_daynum = data_all.replace(data_all_time,data_all_daynum)  

# TODO：我需要更好的替换整列的技巧
# 现在暂时使用删除_添加的方式实现替换整列
del data_all["time"]
data_all["time"] = data_all_daynum   # 替换后的time不再是具体日期，而是天数
data_all.to_csv("0-30data_daynum_int_version.csv")  # 持久化

# 28-30日的数据有特殊作用，要单独取出来
data_28 = data_all[data_all["time"] >= 28]


### 巨型工程来了：
## 需要这么一个数据结构：
## 一张总列表：里面31张子列表；每个子列表里面两个映射表
'''
list
    --list_0
            --dict * 2
    --list_1
            --dict * 2
    ...
    
    --list_30
            --dict * 2
'''

# 总列表，放置所有以天数为单位的列表
useritem_allday_buycnt = []
for idx in range(31):
    # l_day是以天数为基准的列表
    # 两张映射表--> 一张为：人_购买统计,
    #            另一张为：货_被购买投机
    l_day = [{} for i in range(2)]
    data = data_all[data_all["time"] == idx]   # 布尔索引：以daytime为基准
    print(data)
    for dt_idx in data.index:
        type = int(data["behavior_type"][dt_idx]) - 1
        if type == 3:    # 说明是购买行为
            print("buy")
            user = data["user_id"][dt_idx]
            item = data["item_id"][dt_idx]
            if user in l_day[0]:
                l_day[0][user] += 1
            else:
                l_day[0][user] = 1
            if item in l_day[1]:
                l_day[1][item] += 1
            else:
                l_day[1][item] = 1
        else:           # 说明不是购买行为，直接跳过，简化运算
            continue
    useritem_allday_buycnt.append(l_day)    



### 4.解析出29-31日的候选对象uid 

# 29-31日的候选对象uid列表
uid_29_to_31 = []
for row_index in data_28.index:
    user = data_28["user_id"][row_index]
    item = data_28["item_id"][row_index]
    time = int(data_28["time"][row_index]) + 1  # 注意这步的处理：我们的目标是29日的候选对象，所以time要加1
                                                #（是否能成为29日的候选对象，取决于在28日(user,item)这对pair是否产生了behavior）
    uid = (user,item,time) 
    uid_29_to_31.append(uid)
uid_29_to_31 = list(set(uid_train_29))    # 去重之后的 29日训练候选对象列表   
pickle.dump(uid_29_to_31,open("uid_29_to_31.pkl","wb"))  # 序列化

### 5. 分别将29,30,31日的候选对象 从 29-31日的候选对象uid列表中分离出来
info("开始")
uid_train_29  = []
uid_offline_candidate_30 = []
uid_online_candidate_31 = []
for uid in uid_29_to_31:
    if(uid[-1] == 29):
        uid_train_29.append(uid)
    if(uid[-1] == 30):
        uid_offline_candidate_30.append(uid)
    if(uid[-1] == 31):
        uid_online_candidate_31.append(uid)
info("结束")

# 序列化
pickle.dump(uid_train_29,open("uid_train_29.pkl","wb"))
pickle.dump(uid_train_30,open("uid_train_30.pkl","wb"))
pickle.dump(uid_train_31,open("uid_train_31.pkl","wb"))

### 6. 开始针对29日的训练候选对象进行处理

## 6.1 feature: 

# 先对28日的行为次数进行汇总
# 建立4张映射表,分别对应4种行为。表中添加uid与次数的映射
feature_dict = [{} for _ in range(4)]

info("feature映射表开始")
for idx in data_28.index:
    user = data_28["user_id"][idx]
    item = data_28["item_id"][idx]
    daynum = data_28["time"][idx]
    uid = (user,item,daynum)
    type = int(data_28["behavior_type"][idx]) - 1    # 这里-1的原因是因为 feature_dict列表的索引从0开始，要与behavior的1对齐
    if uid in feature_dict[type]:
        feature_dict[type][uid] += 1
    else:
        feature_dict[type][uid] = 1
info("feature映射表结束")

pickle.dump(feature_dict,open("feature_dict.pkl","wb"))

## 6.2 label:

# 注意uid_buy不是最后的标签，而是标签的参考
uid_buy = {}

info("开始")
for idx in data_28.index:
    user = data_28["user_id"][idx]
    item = data_28["item_id"][idx]
    daynum = data_28["time"][idx]
    uid = (user,item,daynum)
    type = int(data_28["behavior_type"][idx])    # 这里的type不用-1
    
    # 上面的逻辑和feature差不多，只有下面这段有区别
    # 【技巧】没有购买的情况没有必要标注为0，因为这不是最终的标签，所以标注0不但费时还没意义，反而会让之后打标签的逻辑变繁琐
    if(type == 4):
        uid_buy[uid] = 1
info("结束")


### 7. 对29日的候选对象进行迭代：拼接每一个对象的特征向量和标签

# 候选对象登场： uid_train_29
# X_train，Y_train属于训练集
# X_train为特征向量集，Y_train为标签集
X_train = np.zeros((len(uid_train_29),4))
Y_train = np.zeros((len(uid_train_29),))    # Y最好构造成1维数组(len,)，而不是N*1的2维数组(len,1)，不然后面索引会有麻烦
feature_label(X_train,Y_train,uid_train_29)


### 8.对30日的候选对象进行特征向量拼接和打标签（操作同29日）

# X_offline和Y_offline是线下测试集
X_offline = np.zeros((len(uid_offline_candidate_30),4))
Y_offline = np.zeros((len(uid_offline_candidate_30),))   
feature_label(X_offline,Y_offline,uid_offline_candidate_30)


### 9. 训练模型
from sklearn.linear_model import LogisticRegression
model_logReg = LogisticRegression()
model_logReg.fit(X_train,Y_train)

Y_offline_prediction = model_logReg.predict_proba(X_offline)

# 30日的预测购买几率
Y_offline_prediction_buy = []   
for proba in Y_offline_prediction:
    Y_offline_prediction_buy.append(proba[1])

# 将30日的预测概率值<score>与实际值进行打包
lx = zip(uid_offline_candidate_30,Y_offline_prediction_buy)

# 按置信度<购买概率>降序排列
lx = sorted(lx,key=lambda x:x[1],reverse=True)

### 10.得到预测结果，提取（user，item）这对pair

# 将概率最高的N_bug个定为成交
# 将pair持久化
N_buy = 2500
with open("offline30_prediction_buy_pair.csv","w") as f:
    for i in range(N_buy):
        item = lx[i]
        f.write("%s,%s\n" % (item[0][0],item[0][1]))

### 11. 获取真实结果
with open("day30_buy_pair.csv","w") as f:
    for idx,buy in enumerate(Y_offline):
        if(buy == 1):
            day30_buy_pair = uid_offline_candidate_30[idx]
            f.write("%s,%s\n" % (day30_buy_pair[0],day30_buy_pair[1]))

### 12. 有了预测结果和真实结果，便可以通过getscore.py来计算F1，P，R指标了。



#-------------------------------------------------------------------------

# 下面是线上评测结果集的获取，操作方法和上面是一样的

X_online = np.zeros((len(uid_online_candidate_31),4))
Y_online = np.zeros((len(uid_online_candidate_31),))   
feature_label(X_online,Y_online,uid_online_candidate_31)  # 调用特征_标签函数

Y_online_prediction = model_logReg.predict_proba(X_online)
Y_online_prediction_buy = []
for proba in Y_online_prediction:
    Y_online_prediction_buy.append(proba[1])

lx_online = zip(uid_online_candidate_31,Y_online_prediction_buy)
lx_online = sorted(lx_online,key=lambda x:x[1],reverse=True)

N_buy = 2000

### 该文件便可作为结果进行提交
with open("tianchi_mobile_recommendation_predict.csv","w") as f:
    for i in range(N_buy):
        item = lx_online[i]
        f.write("%s,%s\n" % (item[0][0],item[0][1]))

