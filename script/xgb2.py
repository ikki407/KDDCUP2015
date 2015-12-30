# -*- coding: UTF-8 -*-

###################################
#feature engineering
#Object.csv使用
#xgboost
###################################
import hashlib
import argparse, csv, sys, pickle, collections, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
from collections import Counter


#データの読み込み
#print 'leading data'
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
enr_train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/enrollment_train.csv')
enr_test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/enrollment_test.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']

len_enr_train = len(enr_train)
len_enr_test = len(enr_test)

#trainにdropのlabelをつける
enr_train = enr_train.merge(label, how='inner',on='enrollment_id')#ここで1つデータが落ちている

#timeをyear, month, day, hourに変換する
f_y = lambda x: x[:4]#year
train['year'] = train['time'].apply(f_y)
test['year'] = test['time'].apply(f_y)

f_m = lambda x: x[5:7]#month
train['month'] = train['time'].apply(f_m)
test['month'] = test['time'].apply(f_m)

f_d = lambda x: x[8:10]#day
train['day'] = train['time'].apply(f_d)
test['day'] = test['time'].apply(f_d)

f_h = lambda x: x[11:13]#hour
train['hour'] = train['time'].apply(f_h)
test['hour'] = test['time'].apply(f_h)

del train['time']
del test['time']

#enr_trainの中のusernameとcourse_idのカウント数を各idに追加する
X = pd.concat((enr_train[[col for col in enr_train.columns if col in ["enrollment_id",'username','course_id'] ]], enr_test))
count_lk = {col: X.groupby(col).aggregate({"enrollment_id": "count"}).to_dict()["enrollment_id"] for col in X.columns}

for col in ['username','course_id']:
    enr_train[col + "_count"] = ([count_lk[col][x] for x in enr_train[col]])
    enr_test[col + "_count"] = ([count_lk[col][x] for x in enr_test[col]])


#enr_trainにlog_trainのデータを付加する
for col in ['source','event','year','month','day','hour']:
    l_train_count = train.groupby(['enrollment_id',col]).aggregate({'enrollment_id':'count'}).unstack()
    l_train_count['enrollment_id']['enrollment_id'] = l_train_count['enrollment_id'].index

    col_count = l_train_count['enrollment_id']
    col_count.index = enr_train.index
    if col == 'month':
        col_count.columns = col_count.columns + '_m'
    if col == 'day':
        col_count.columns = col_count.columns + '_d'
    if col == 'hour':
        col_count.columns = col_count.columns + '_h'
    enr_train = pd.concat([enr_train,col_count],axis=1)

#log_trainでの各idの連続する数を追加する
enr_train['sum_count'] = enr_train[['server','browser']].sum(1)
enr_train = enr_train.fillna(0)


#enr_testにlog_testのデータを付加する
for col in ['source','event','year','month','day','hour']:
    l_test_count = test.groupby(['enrollment_id',col]).aggregate({'enrollment_id':'count'}).unstack()
    l_test_count['enrollment_id']['enrollment_id'] = l_test_count['enrollment_id'].index

    col_count = l_test_count['enrollment_id']
    col_count.index = enr_test.index
    if col == 'month':
        col_count.columns = col_count.columns + '_m'
    if col == 'day':
        col_count.columns = col_count.columns + '_d'
    if col == 'hour':
        col_count.columns = col_count.columns + '_h'
    enr_test = pd.concat([enr_test,col_count],axis=1)

#log_testでの各idの連続する数を追加する
enr_test['sum_count'] = enr_test[['server','browser']].sum(1)
enr_test = enr_test.fillna(0)

#Objectのカウント
#cnt = Counter()
#tr_te = pd.concat([train,test])
#for i in tr_te['object']:
#    cnt[i]+=1
#
#obj_count = pd.DataFrame(cnt.items(),columns=['object','object_count'])
#train = train.merge(obj_count,on='object',how='outer')
#test = test.merge(obj_count,on='object',how='inner')
#enr_train = enr_train.merge(train[['username','object_count']],on='username',how='inner')



#object.csv
Object = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/object.csv')

#enr_trainの中のusernameとcourse_idのカウント数を各idに追加する
#X = pd.concat((Object[[col for col in Object.columns if col in ["course_id"] ]], enr_test))
count_lk = {col: Object.groupby(col).aggregate({"course_id": "count"}).to_dict()["course_id"] for col in Object.columns}

for col in ["course_id"]:
    enr_train[col + "_count_obj"] = ([count_lk[col][x] for x in enr_train[col]])
    enr_test[col + "_count_obj"] = ([count_lk[col][x] for x in enr_test[col]])

for col in ['category']:
    l_train_count = Object.groupby(['course_id',col]).aggregate({'course_id':'count'}).unstack()
    l_train_count['course_id']['course_id'] = l_train_count['course_id'].index

    col_count = l_train_count['course_id']
    #col_count.index = Object.index
    col_count.columns = col_count.columns + '_obj'
    col_count = col_count.fillna(0)
    #enr_train = pd.concat([enr_train,col_count],axis=1)

course_id_label = pd.DataFrame(col_count.index, index = range(0,len(col_count.index)))
course_id_count_obj = col_count
course_id_count_obj.index = range(0,len(col_count.index))
new_object = pd.concat([course_id_label, course_id_count_obj],axis=1)
#trainに付加
enr_train = enr_train.merge(new_object, how='inner',on='course_id', sort=False)

#testに付加
enr_test = enr_test.merge(new_object, how='inner',on='course_id', sort=False)


X = pd.concat((train, test))
count_lk = {col: X.groupby(col).aggregate({"enrollment_id": "count"}).to_dict()["enrollment_id"] for col in X.columns}

object_count = pd.DataFrame(count_lk[col].items())
object_count.columns = ['object', 'obj_count']


enr_train.merge(object_count, how='inner',on='object', sort=False)



for col in ['object']:
    enr_train[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    enr_test[col + "_count"] = ([count_lk[col][x] for x in test[col]])



b = categorical(np.array(train['object']), drop=True)
b = pd.DataFrame(b)




#enr_trainvalを1:2にsplit
n_val = round(len(enr_train)*0.2)
rows = np.random.choice(enr_train.index.values, len(enr_train))

new_train = enr_train.ix[rows[n_val:]]
val_data = enr_train.ix[rows[:n_val]]
shuffle_train = enr_train.ix[rows]

val_label = val_data['drop']

print '(new_train+val_data), shuffle_trian, enr_testが元の長さと同じか'
print 'dropデータの無いtrainの中の1つのデータが落ちたから1少ない'

print (len(new_train)+len(val_data))==len_enr_train-1, len(shuffle_train) == len_enr_train-1, len(enr_test) == len_enr_test, len(val_label) == len(val_data)

print new_train.columns
print enr_test.columns
print new_train.columns == val_data.columns
print new_train.columns == shuffle_train.columns



new_train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/new_enr_train.csv',index=False)
val_data.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/enr_validation.csv',index=False)
shuffle_train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train.csv',index=False)
enr_test.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv',index=False)
val_label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/val_enr_label.csv',index=False)


