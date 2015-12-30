import pandas as pd
from datetime import datetime
from csv import DictReader
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
import matplotlib.pyplot as plt
import math
import xgboost as xgb
import random
from sklearn.feature_extraction import DictVectorizer as DV
from scikits.statsmodels.tools import categorical
from datetime import date
from collections import Counter
from sklearn.ensemble import BaggingClassifier


#データの読み込み
#print 'leading data'
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train_depth.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test_depth.csv')
train = train.fillna(0)
test = test.fillna(0)
f_tostr = lambda x: str(x)
train['module_depth'] = train['module_depth'].apply(f_tostr)
test['module_depth'] = test['module_depth'].apply(f_tostr)

'''
#log_trainをduplicateする
time_pre = train['time']#timeの一時保存
del train['time']
print len(train)
train = train.drop_duplicates()
print len(train)
train = train.join(time_pre,how='left')#trainにtimeを戻す

time_pre = test['time']#timeの一時保存
del test['time']
print len(test)
test = test.drop_duplicates()
print len(test)
test = test.join(time_pre,how='left')#trainにtimeを戻す
'''
'''
#dropの時系列を見る
f_time = lambda x: date(int(x[:4]),int(x[5:7]),int(x[8:10])).toordinal()
train['time'] = train['time'].apply(f_time)
test['time'] = test['time'].apply(f_time)

train['time'] = train['time'] - 735000
test['time'] = test['time'] - 735000

X = pd.concat([train,test])
train = train.merge(label,on='enrollment_id',how='inner')
train = train.sort('time')

test = test.sort('time')

#a = train.drop_duplicates('enrollment_id')
#b = test.drop_duplicates('enrollment_id')


a = dtrain
a['min_time'] = sorted(a['min_time'])
a['drop'] = label
b = dtest
b['min_time'] = sorted(b['min_time'])

#plt.plot(a['time'].values, a['drop'].values.cumsum())
#plt.show()
c = np.ones(len(a))

plt.figure()
plt.hist(b['min_time'].values,bins=300,alpha=0.5,label = "Test")
plt.hist(a['min_time'].values,bins=300,alpha=0.5,label = "Train")
plt.xlabel('time')
plt.ylabel('participant')
plt.legend()
ax = plt.twinx()
ax.plot(a['min_time'].values, a['drop'].values.cumsum())
ax.plot(a['min_time'].values, np.cumsum(c))

plt.ylabel('cumsum of dropout in train')
plt.show()

X = X.sort('time')
'''
label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
enr_train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/enrollment_train.csv')
enr_test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/enrollment_test.csv')
'''
#objectの各mojuleの木構造の深さを作成
Object = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/object.csv')
Object = Object.fillna('0')
Object = Object.drop_duplicates()
Object.index = range(0,len(Object))
Object['module_depth'] = pd.DataFrame(np.zeros(len(Object)))
for i in Object.index:
    obj = Object.iloc[i,:]['module_id']
    depth = 1
    while True:
        if len(Object.iloc[Object['children'].str.contains(obj).values,:]) >= 1:
            #print (Object.iloc[Object['children'].str.contains(obj).values,:])
            obj = Object.iloc[Object['children'].str.contains(obj).values,:]['module_id'].values[0]
            depth += 1
        else:
            Object.loc[i,'module_depth'] = depth
            break
    print i,depth

Object.columns = [u'course_id', u'object', u'category', u'children', u'start', u'module_depth']
yy = Object[[u'object',u'module_depth']]
Object.to_csv('/Users/IkkiTanaka/Documents/KDDCup/Object_depth.csv',index=False)
train = train.merge(yy ,how='left',on='object')
test = test.merge(yy,how='left',on='object')

train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train_depth.csv',index=False)
test.to_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test_depth.csv',index=False)
'''
'''
#Objectのcategoryを追加する
Object = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/object.csv')
Object = Object.fillna('0')
Object = Object.drop_duplicates()
Object.index = range(0,len(Object))

Object.columns = [u'course_id', u'object', u'category', u'children', u'start']
yy = Object[[u'object',u'category']]
train = train.merge(yy ,how='left',on='object')
test = test.merge(yy,how='left',on='object')

train = train.fillna('NaN')
test = test.fillna('NaN')

'''


len_enr_train = len(enr_train)
len_enr_test = len(enr_test)

#trainにdropのlabelをつける
enr_train = enr_train.merge(label, how='inner',on='enrollment_id')#ここで1つデータが落ちている
#enrollment_id = 139669

X = pd.concat([train,test])
#X.index = range(0,len(X))
X['object_fact'] = pd.factorize(X['object'])[0]#objectのFactorize化
train['object_fact'] = X['object_fact'][:len(train)]
test['object_fact'] = X['object_fact'][len(train):]

object_fact_train = train.groupby(['enrollment_id'])['object_fact'].mean()
object_fact_train.index = enr_train.index

object_fact_test = test.groupby(['enrollment_id'])['object_fact'].mean()
object_fact_test.index = enr_test.index

enr_train['object_fact'] = object_fact_train
enr_test['object_fact'] = object_fact_test


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

#曜日をつける
f_w = lambda x: str(date(int(x[:4]),int(x[5:7]),int(x[8:10])).weekday())
train['weekday'] = train['time'].apply(f_w)
test['weekday'] = test['time'].apply(f_w)

del train['time']
del test['time']

'''
f_time = lambda x: date(int(x[:4]),int(x[5:7]),int(x[8:10])).toordinal()
train['time'] = train['time'].apply(f_time)
test['time'] = test['time'].apply(f_time)

train['time'] = train['time'] - 735000
test['time'] = test['time'] - 735000
'''

#LibFFM用
#n_val = round(len(train)*0.2)
#rows = np.random.choice(train.index.values, len(train),replace=False)
#train = train.ix[rows]
#train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_fm.csv', index=False)
#test.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv', index=False)


#enr_trainの中のusernameとcourse_idのカウント数を各idに追加する
X = pd.concat((enr_train[[u'enrollment_id', u'username', u'course_id']], enr_test),ignore_index=True)

count_lk = {col: X.groupby(col).aggregate({"enrollment_id": "count"}).to_dict()["enrollment_id"] for col in X.columns}

for col in ['username','course_id']:
    enr_train[col + "_count"] = ([count_lk[col][x] for x in enr_train[col]])
    enr_test[col + "_count"] = ([count_lk[col][x] for x in enr_test[col]])


#enr_trainにlog_trainのデータを付加する
for col in ['source','event','year','month','day','hour','weekday',u'module_depth']:#'source',
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
    if col == 'weekday':
        col_count.columns = col_count.columns + '_w'
    if col == 'module_depth':
        col_count.columns = col_count.columns + '_md'
    if col == 'category':
        col_count.columns = col_count.columns + '_cg'
    enr_train = pd.concat([enr_train,col_count],axis=1)

Object = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/object.csv')
Object = Object.fillna('0')
Object = Object.drop_duplicates()
Object.index = range(0,len(Object))

#enr_trainにObjectのデータを付加する
for col in ['category']:#'source',
    l_train_count = Object.groupby(['course_id',col]).aggregate({'course_id':'count'}).unstack()
    l_train_count['course_id']['course_id'] = l_train_count['course_id'].index

    col_count = l_train_count['course_id']
    #col_count.index = enr_train.index
    if col == 'category':
        col_count.columns = col_count.columns + '_cg'
    col_count['course_id']=col_count.index
    enr_train = enr_train.merge(col_count,on='course_id',how='left')




#log_trainでの各idの連続する数を追加する
enr_train['sum_count'] = enr_train[['server','browser']].sum(1)
enr_train = enr_train.fillna(0)


#enr_testにlog_testのデータを付加する
for col in ['source','event','year','month','day','hour','weekday',u'module_depth']:
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
    if col == 'weekday':
        col_count.columns = col_count.columns + '_w'
    if col == 'module_depth':
        col_count.columns = col_count.columns + '_md'
    if col == 'category':
        col_count.columns = col_count.columns + '_cg'

    enr_test = pd.concat([enr_test,col_count],axis=1)

#enr_trainにObjectのデータを付加する
for col in ['category']:#'source',
    l_test_count = Object.groupby(['course_id',col]).aggregate({'course_id':'count'}).unstack()
    l_test_count['course_id']['course_id'] = l_test_count['course_id'].index

    col_count = l_test_count['course_id']
    #col_count.index = enr_train.index
    if col == 'category':
        col_count.columns = col_count.columns + '_cg'
    col_count['course_id']=col_count.index
    enr_test = enr_test.merge(col_count,on='course_id',how='left')


#log_testでの各idの連続する数を追加する
enr_test['sum_count'] = enr_test[['server','browser']].sum(1)
enr_test = enr_test.fillna(0)

#del enr_train['server'],enr_train['browser']
#del enr_test['server'],enr_test['browser']

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
#Object = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/object.csv')
#Object['module_id'].columns = ['object']
#Object.merge(enr_train, on='object', how='inner')[['category','children']]
#Objectのcource_idの情報を追加

#time幅追加
#enr_train['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
#enr_test['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')



#enr_trainvalを1:2にsplit
n_val = round(len(enr_train)*0.2)
np.random.seed(19920407)
rows = np.random.choice(enr_train.index.values, len(enr_train),replace=False)

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
shuffle_train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train_depth.csv',index=False)
enr_test.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test_depth.csv',index=False)
val_label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/val_enr_label_depth.csv',index=False)




#log transform
f = lambda x: np.log(1+x**2) if x > 1 else x



dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train_depth.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test_depth.csv')


label = dtrain['drop']
#del dtrain['drop']

X = pd.concat([dtrain,dtest])
X.index = range(0,len(X))

#course_idのダミー変数化
#b = categorical(np.array(X['course_id']), drop=True)
#b = pd.DataFrame(b)

#aaa = pd.factorize(X['username'])#usernameのFactorize化
#dtrain['username'] = aaa[0][:len(dtrain)]
#dtrain_leaf[dtrain_leaf['username_count'] <= 5]['username'] = -999

#X = pd.concat([train,test])
#X.index = range(0,len(X))
#bbb = pd.factorize(X['course_id'])#course_idのFactorize化
#dtrain['course_id'] = bbb[0][:len(dtrain)]



del dtrain['enrollment_id'],#dtrain['username'],#dtrain['course_id'],
#dtrain = pd.concat([b[:len(dtrain)],dtrain],axis=1)
#label = dtrain['drop']
#del dtrain['drop']
#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換


#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv')
#b = categorical(np.array(dtest['course_id']), drop=True)
#b = pd.DataFrame(b)
#dtest['username'] = aaa[0][len(dtrain):]

#dtest['course_id'] = bbb[0][len(dtrain):]



del dtest['enrollment_id'],#dtest['username'], #dtest['course_id']

#xx = b[len(dtrain):]
#xx.index = range(0,len(xx))
#dtest = pd.concat([xx,dtest],axis=1)
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換

#year, month, day, hour ,weekdayのカウントデータを1にする
f_to1 = lambda x: 1 if x >=1 else 0
label_to1 = [u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w']

dtrain[label_to1] = dtrain[label_to1].applymap(f_to1)
dtest[label_to1] = dtest[label_to1].applymap(f_to1)

'''
#sum_countで割って, percentageを追加する
for col in [u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki']:
    dtrain[col+'_pct'] = dtrain[col]/dtrain[u'sum_count']

for col in [u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki']:
    dtest[col+'_pct'] = dtest[col]/dtest[u'sum_count']
'''


#dtrain.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_NN.csv',index=False)
#dtest.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_NN.csv',index=False)
'''
#log_trainのなかの各idの最初と最後の時間を抽出
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')

f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10]))
train_id_time = train[['enrollment_id','time']]
test_id_time = test[['enrollment_id','time']]
train_id_time['time'] = train_id_time['time'].apply(f_time)
test_id_time['time'] = test_id_time['time'].apply(f_time)

train_time = pd.DataFrame(np.zeros(len(dtrain)),columns=['time'])
i = 0
a = train_id_time['enrollment_id'].drop_duplicates().values
for id in a:
    train_time_ = train_id_time.iloc[(train_id_time['enrollment_id']==id).values,1]
    train_time.iloc[i,:] = (max(train_time_)-min(train_time_)).days
    i+=1
    if i % 1000 == 0:
        print i

test_time = pd.DataFrame(np.zeros(len(dtest)),columns=['time'])
i = 0
a = test_id_time['enrollment_id'].drop_duplicates().values
for id in a:
    test_time_ = test_id_time.iloc[(test_id_time['enrollment_id']==id).values,1]
    test_time.iloc[i,:] = (max(test_time_)-min(test_time_)).days
    i+=1
    if i % 1000 == 0:
        print i

dtrain['time_len'] = train_time
dtest['time_len'] = test_time
#train_timelenを読み込んでdtrainに付け加える
train_time.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv',index=False)
test_time.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv',index=False)

#dtrain['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
#dtest['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')

'''

#train_course = dtrain['course_id'] +1
#test_course = dtest['course_id'] +1


#dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')

#dtrain = dtrain.iloc[:,39:]
#dtest = dtest.iloc[:,39:]

#0の値を-1にする
#f_to0 = lambda x: -1 if x == 0 else x
#dtrain = dtrain.applymap(f_to0)
#dtest = dtest.applymap(f_to0)


aa = dtrain.columns
#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換

dtrain[['min_time','max_time','len_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
dtest[['min_time','max_time','len_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')

dtrain_leaf = dtrain
dtest_leaf = dtest
'''
dtrain_leaf.loc[dtrain_leaf['username_count'] <= 5,['username']] = -999
dtest_leaf.loc[dtest_leaf['username_count'] <= 5,['username']] = -999

dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_NN.csv',index=False)
dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/label_NN.csv',index=False)

dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/NN/train_NN.csv',index=False)
dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/label_NN.csv',index=False)

dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/train_NN.csv',index=False)
dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/label_NN.csv',index=False)


'''
dtrain_leaf['drop'] = label
dtrain_leaf = dtrain_leaf.sort('min_time')
dtrain_leaf['cumsum_drop'] = np.cumsum(dtrain_leaf['drop'])
label2 = dtrain_leaf['drop']

del dtrain_leaf['drop']
dtrain = xgb.DMatrix(dtrain_leaf, label=label2)

#dtrain_leaf['course_id'] = bbb[0][:len(dtrain)]
#dtest_leaf['course_id'] = bbb[0][len(dtrain):]
'''
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, u'username_count', u'course_id_count', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'sum_count']
'''
#using_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, u'username_count', u'course_id_count', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w']
#days = [u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d']
#for day in days:
#    del dtrain_leaf[day], dtest_leaf[day]
#dtrain = dtrain[using_label]
#dtest = dtest[using_label]
'''
#LibFFM用
#dtrain['drop'] = label['drop']
n_val = round(len(dtrain)*0.2)
rows = np.random.choice(dtrain.index.values, len(dtrain),replace=False)
train_fm = dtrain.ix[rows[n_val:]]
val_fm = dtrain.ix[rows[:n_val]]
#del dtrain['drop']
train_fm.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_fm.csv', index=False)
val_fm.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/enr_validation_fm.csv', index=False)
dtest.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv', index=False)


'''




print dtrain_leaf.columns == dtest_leaf.columns

dtrain = xgb.DMatrix(dtrain, label=label)
dtest = xgb.DMatrix(dtest)


def label_selection(random_state = 19920407, min_variable = 5):
    course_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    username_count = [u'username_count']
    couse_id_count = [u'course_id_count']
    source = [u'browser', u'server']
    event = [u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki']
    year = [u'2013', u'2014']
    month = [u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m']
    day = [u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d']
    hour = [u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h']
    weekday = [u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w']
    sum_count = [u'sum_count']
    time_len = [u'time_len']
    module_depth = [u'0.0_md', u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md']
    np.random.seed(random_state)
    number_var = np.random.randint(min_variable,14)
    variable = [course_id,username_count,couse_id_count,source,event,year,month,day,hour,weekday,sum_count,time_len,module_depth]
    selected_variable = np.random.choice(variable,number_var,replace=False)
    using_variable = []
    for i in xrange(0,len(selected_variable)):
        using_variable += selected_variable[i]
    print 'number of variables: ', number_var
    print 'variables: ', using_variable
    return using_variable



#feature_importance
def Feature_importance(bst, aa):
    a = bst.get_fscore().items()
    fi = pd.DataFrame(a).sort(0,ascending=True)
    f = lambda x: int(x.replace('f',''))
    fi.iloc[:,0] = fi.iloc[:,0].apply(f)
    fi = fi.sort(0,ascending=True)

    aaa = pd.DataFrame(aa)
    aaa[1] = range(0,len(aa))
    aaa.columns = [1,0]
    #fi.iloc[:,0] = aa
    #fi = fi.sort_index(1,ascending=False)
    fi = fi.merge(aaa,on=0,how='outer').fillna(0)[['1_y','1_x']]
    #fi[0] = fi.index
    #fi[0] = aaa[1]
    fi.columns = [0,1]
    fi = fi.sort(1,ascending=False)
    fi.index = range(0, len(fi))

    #１個ずれてる
    plt.bar(range(len(fi[1])), fi[1], align='center')
    plt.xticks(range(len(fi[1])), fi[0], size='small')
    plt.show()





label_spw = dtrain.get_label()
scale_pos_weight = float(np.sum(label_spw == 0)) / np.sum(label_spw==1)

#scale_pos_weight =0.2
pred_name = []
num_model = 500
preds = pd.DataFrame()
#dtrain = xgb.DMatrix(dtrain_leaf.loc[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, u'username_count', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m',u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d']], label=label)

for i in xrange(0,num_model):
    random.seed(i*10000)

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(4,4),
              "bst:eta" :  random.uniform(.05, .2),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,4),
              "bst:subsample" :  round(random.uniform(.6, 1.),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.6, 1.),2),#column_subsample
              "silent": 1,
              "base_score": 0.55,
              #"max_delta_step": 10,
              #"scale_pos_weight": scale_pos_weight,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(300, 350),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    watchlist  = [(dtrain,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,watchlist)

    bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    bst = pd.Series(bst).apply(f_)
    num_round_ = bst.argmax() + 35
    cc = bst.max()
    if True:#cc > 0.885:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob1 = bst.predict( dtrain )
        pred_prob2 = bst.predict( dtest )
        
        dtrain2 = xgb.DMatrix(pd.concat([dtrain_leaf,pd.DataFrame(pred_prob1)],ignore_index=True,axis=1), label=label)
        dtest2 = xgb.DMatrix(pd.concat([dtest_leaf,pd.DataFrame(pred_prob2)],ignore_index=True,axis=1))

        param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(4,4),
              "bst:eta" :  random.uniform(.05, .2),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,4),
              "bst:subsample" :  round(random.uniform(.7, 1.),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, 1.),2),#column_subsample
              "silent": 1,
              #"base_score": 0.55,
              #"max_delta_step": 10,
              #"scale_pos_weight": scale_pos_weight,
              }
        #if random.randint(0,1):
        #    param['scale_pos_weight'] = scale_pos_weight
        num_round = int( round(random.uniform(160, 180),0))
        #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
        print param
        print num_round

        bst=xgb.cv(param, dtrain2, num_round, nfold=5, seed = 19920407, show_stdv = True)
        
        watchlist  = [(dtrain2,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain2, num_round,watchlist)
        pred_prob = bst.predict( dtest2 )


        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred30.csv',header=None,index=False)



pred_name = []
num_model = 300
for i in xrange(0,num_model):
    #using_label2 = label_selection(random_state = i, min_variable = 5)
    dtrain_leaf_resample, label_resample =bootstrap_resample(dtrain_leaf, n=.5,random_state=i)
    dtrain = xgb.DMatrix(dtrain_leaf_resample, label=label_resample)

    #print using_label2
    #dtrain2 = xgb.DMatrix(dtrain_leaf[using_label2], label=label)
    #dtest2 = xgb.DMatrix(dtest_leaf[using_label2])

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(4,4),
              "bst:eta" :  round(random.uniform(.01, .3),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 3.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  0.3,#round(random.uniform(.2, 1.0),2),#row_subsample
              "colsample_bytree" :  1,#round(random.uniform(.2, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(300, 450),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    watchlist  = [(dtrain,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,watchlist)

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    #num_round_ = bst.argmax()
    #cc = bst.max()
    if True:#cc > 0.883:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred32.csv',header=None,index=False)


Feature_importance(bst, aa)

def bootstrap_resample(X, n=1.0, random_state=407):
    X['drop'] = label
    n_val = round(len(X))
    np.random.seed(random_state)
    rows = np.random.choice(X.index.values, len(X)*n,replace=True)
    X_resample = X.ix[rows]
    label_resample = X_resample['drop']
    del X_resample['drop']
    return X_resample, label_resample

#dtrain_leaf_resample, label_resample  =bootstrap_resample(dtrain_leaf,random_state=407)


pred_name = []
num_model = 10
for i in xrange(0,num_model):
    random.seed(i*10000)
    #dtrain_leaf_resample, label_resample =bootstrap_resample(dtrain_leaf, n=.5,random_state=i)
    xxx = len(dtrain_leaf)/2
    #dtrain_leaf_resample = dtrain_leaf.iloc[:xxx,:]
    dtrain1 = xgb.DMatrix(dtrain_leaf.iloc[:xxx,:], label=label[:xxx])
    dtrain2 = xgb.DMatrix(dtrain_leaf.iloc[xxx:,:], label=label[xxx:])
    
    xxxx = len(dtest_leaf)/2
    #dtrain_leaf_resample = dtrain_leaf.iloc[:xxx,:]
    dtest1 = xgb.DMatrix(dtest_leaf.iloc[:xxx,:])
    dtest2 = xgb.DMatrix(dtest_leaf.iloc[xxx:,:])

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" : random.randint(2,3),
              "bst:eta" :  round(random.uniform(.05, .17),2),#step_size
              "bst:gamma" : round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.2, 1.0),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.2, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(250, 300),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    watchlist  = [(dtrain,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,watchlist)

    bst=xgb.cv(param, dtrain1, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    #num_round_ = bst.argmax()
    #cc = bst.max()
    if True:#cc > 0.883:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst1 = xgb.train(param,dtrain1, num_round,watchlist)
        pred_prob1 = bst1.predict( dtest1 )
        bst2 = xgb.train(param,dtrain2, num_round,watchlist)
        pred_prob2 = bst2.predict( dtest2 )
        pred_prob = pd.concat([pd.DataFrame(pred_prob1),pd.DataFrame(pred_prob2)],ignore_index=True)
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pred_prob],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i.csv' % (i,num_round, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred30.csv',header=None,index=False)



def bootstrap_resample(X, n=1.0, random_state=407):
    X['drop'] = label
    n_val = round(len(X))
    np.random.seed(random_state)
    rows = np.random.choice(X.index.values, len(X)*n,replace=False)
    X_resample = X.ix[rows]
    label_resample = X_resample['drop']
    del X_resample['drop']
    return X_resample, label_resample

#dtrain_leaf_resample, label_resample  =bootstrap_resample(dtrain_leaf,random_state=407)



num_model = 10
#preds_tr = pd.DataFrame()
preds_te = dtest_leaf
#xxx = len(dtrain_leaf)/2
#dtrain_leaf_resample = dtrain_leaf.iloc[:xxx,:]
dtrain2 = xgb.DMatrix(dtrain_leaf, label=label)

preds_tr = dtrain_leaf



dtrain_leaf['drop'] = label
xxx1 = dtrain_leaf.iloc[(dtrain_leaf['max_time']>325).values,:]
label_xxx1 = xxx1['drop']
xxx2 = dtrain_leaf.iloc[(dtrain_leaf['max_time']<325).values,:]
label_xxx2 = xxx2['drop']
preds_tr = xxx2
preds_te = dtest_leaf

del xxx1['drop'],xxx2['drop']

dtrain1 = xgb.DMatrix(xxx1,label=label_xxx1)
dtrain2 = xgb.DMatrix(xxx2,label=label_xxx1) 

#dtest_leaf.iloc[(dtest_leaf.loc[:,['max_time']]<350).values),:]


for i in xrange(0,num_model):
    random.seed(i*10000)
    

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(2,2),
              "bst:eta" :  random.uniform(.05, .2),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,4),
              "bst:subsample" :  round(random.uniform(.2, 1.),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.2, 1.),2),#column_subsample
              "silent": 1,
              #"base_score": 0.55,
              #"max_delta_step": 10,
              #"scale_pos_weight": scale_pos_weight,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(800, 1250),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #watchlist  = [(dtrain,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,watchlist)

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    #num_round_ = bst.argmax() + 35
    #cc = bst.max()
    if True:#cc > 0.882:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain1, num_round,watchlist)
        pred_prob1 = bst.predict( dtrain2 )
        pred_prob2 = bst.predict( dtest )
        preds_tr = pd.concat([preds_tr,pd.DataFrame(pred_prob1)],ignore_index=True,axis=1)
        preds_te = pd.concat([preds_te,pd.DataFrame(pred_prob2)],ignore_index=True,axis=1)

preds_tr.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/train_NN.csv',index=False)
preds_te.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/label_NN.csv',index=False)

        dtrain2 = xgb.DMatrix(pd.concat([dtrain_leaf,pd.DataFrame(pred_prob1)],ignore_index=True,axis=1), label=label)
        dtest2 = xgb.DMatrix(pd.concat([dtest_leaf,pd.DataFrame(pred_prob2)],ignore_index=True,axis=1))

        param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(4,4),
              "bst:eta" :  random.uniform(.05, .2),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,4),
              "bst:subsample" :  round(random.uniform(.7, 1.),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, 1.),2),#column_subsample
              "silent": 1,
              #"base_score": 0.55,
              #"max_delta_step": 10,
              #"scale_pos_weight": scale_pos_weight,
              }
        #if random.randint(0,1):
        #    param['scale_pos_weight'] = scale_pos_weight
        num_round = int( round(random.uniform(160, 180),0))
        #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
        print param
        print num_round

        #bst=xgb.cv(param, dtrain2, num_round, nfold=5, seed = 19920407, show_stdv = True)
        
        watchlist  = [(dtrain2,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain2, num_round,watchlist)
        pred_prob = bst.predict( dtest2 )


        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round, param["bst:max_depth"]) ,header=None,index=False)

dtrain = xgb.DMatrix(dtrain_leaf, label=label)
dtest = xgb.DMatrix(dtest_leaf)

pred_name = []
num_model = 50
preds = pd.DataFrame()
for i in xrange(0,num_model):
    random.seed(i*10000)

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  random.uniform(.05, .2),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,4),
              "bst:subsample" :  round(random.uniform(.6, 1.),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.6, 1.),2),#column_subsample
              "silent": 1,
              "base_score": 0.55,
              #"max_delta_step": 10,
              #"scale_pos_weight": scale_pos_weight,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(400, 450),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    watchlist  = [(dtrain,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,watchlist)

    bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    bst = pd.Series(bst).apply(f_)
    num_round_ = bst.argmax() + 35
    cc = bst.max()
    if cc > 0.885:
        watchlist  = [(dtrain,'train')]
        random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i_%f.csv' % (i,num_round, param["bst:max_depth"],cc))
        print preds
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i_%f.csv' % (i,num_round, param["bst:max_depth"],cc) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,len(pred_name)):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(len(pred_name))
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred40.csv',header=None,index=False)


pred_name = []
num_model = 10
preds = pd.DataFrame()
for i in xrange(0,num_model):
    random.seed(i*10000)

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(5,5),
              "bst:eta" :  random.uniform(.01, .1),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,4),
              "bst:subsample" :  round(random.uniform(.6, 1.),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.6, 1.),2),#column_subsample
              "silent": 1,
              #"base_score": 0.55,
              #"max_delta_step": 10,
              "scale_pos_weight": scale_pos_weight,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(400, 450),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    watchlist  = [(dtrain,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,watchlist)

    bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    bst = pd.Series(bst).apply(f_)
    num_round_ = bst.argmax() + 35
    cc = bst.max()
    if cc > 0.885:
        watchlist  = [(dtrain,'train')]
        random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i_%f.csv' % (i,num_round_, param["bst:max_depth"],cc))
        print preds
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i_%f.csv' % (i,num_round_, param["bst:max_depth"],cc) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,len(pred_name)):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(len(pred_name))
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred41.csv',header=None,index=False)


