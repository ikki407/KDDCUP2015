####
#log_train.csv
####

'''
1. trainをsplitする8:2

train: new_train.csv
validation: validation.csv -> label: val_label

2. all_dataはtimeを変換して
shuffle_train.csv
new_test

'''
import hashlib
import argparse, csv, sys, pickle, collections, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
from collections import Counter
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
enr_train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/enrollment_train.csv')
enr_test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/enrollment_test.csv')

train = train.merge(label, how='inner',on='enrollment_id')
print len(train)
f = lambda x: x[:10]
train['time'] = train['time'].apply(f)
test['time'] = test['time'].apply(f)

train = train.drop_duplicates()
test = test.drop_duplicates()
print len(train)

#del train['time']
#時間はhourも見たほうが良いかも！？


X = pd.concat((train[[col for col in train.columns if col in ["enrollment_id",'username','course_id','time','source','event','object'] ]], test))
count_lk = {col: X.groupby(col).aggregate({"enrollment_id": "count"}).to_dict()["enrollment_id"] for col in X.columns}

for col in ["enrollment_id",'course_id','source','event','object']:
    train[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    test[col + "_count"] = ([count_lk[col][x] for x in test[col]])

cnt = Counter()
enr_tr_te = pd.concat([enr_train,enr_test])

for i in enr_tr_te['username']:
    cnt[i]+=1

usr_count = pd.DataFrame(cnt.items(),columns=['username','username_count'])

train = train.merge(usr_count,on='username',how='inner')
test = test.merge(usr_count,on='username',how='inner')
print len(train)

for col in ['source','event']:
    l_test_count = test.groupby(['enrollment_id',col]).aggregate({'enrollment_id':'count'}).unstack()
    l_test_count['enrollment_id']['enrollment_id'] = l_test_count['enrollment_id'].index

    col_count = l_test_count['enrollment_id']
    col_count.index = enr_test.index
    enr_test = pd.concat([enr_test,col_count],axis=1)

enr_test['sum_count'] = enr_test[['server','browser']].sum(1)
enr_test = enr_test.fillna(0)

n_val = round(len(train)*0.2)
rows = np.random.choice(train.index.values, len(train))

new_train = train.ix[rows[n_val:]]
val_data = train.ix[rows[:n_val]]
shuffle_train = train.ix[rows]

val_label = val_data['drop']


new_train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/train/new_train.csv',index=False)
val_data.to_csv('/Users/IkkiTanaka/Documents/KDDCup/train/validation.csv',index=False)
shuffle_train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/train/shuffle_train.csv',index=False)
test.to_csv('/Users/IkkiTanaka/Documents/KDDCup/test/new_test.csv',index=False)

val_label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/train/val_label.csv',index=False)



