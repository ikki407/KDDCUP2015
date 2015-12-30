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



#データの読み込み
#print 'leading data'
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train_depth.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test_depth.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
enr_train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/enrollment_train.csv')
enr_test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/enrollment_test.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
train = train.fillna(0)
test = test.fillna(0)
f_tostr = lambda x: str(x)
train['module_depth'] = train['module_depth'].apply(f_tostr)
test['module_depth'] = test['module_depth'].apply(f_tostr)




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
        if len(dtObject.iloc[Object['children'].str.contains(obj).values,:]) >= 1:
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
len_enr_train = len(enr_train)
len_enr_test = len(enr_test)

Object = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/object.csv')
Object = Object.fillna('0')
Object = Object.drop_duplicates()
Object.index = range(0,len(Object))

train['category'] = train.merge(Object,how='left',left_on='object',right_on='module_id')['category']
train = train.fillna('0')
test['category'] = test.merge(Object,how='left',left_on='object',right_on='module_id')['category']
test = test.fillna('0')

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

#曜日をつける
f_w = lambda x: str(date(int(x[:4]),int(x[5:7]),int(x[8:10])).weekday())
train['weekday'] = train['time'].apply(f_w)
test['weekday'] = test['time'].apply(f_w)

del train['time']
del test['time']

#LibFFM用
#n_val = round(len(train)*0.2)
#rows = np.random.choice(train.index.values, len(train),replace=False)
#train = train.ix[rows]
#train = train.drop_duplicates()
#test = test.drop_duplicates()

#train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_fm.csv', index=False)
#test.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv', index=False)


#enr_trainの中のusernameとcourse_idのカウント数を各idに追加する
X = pd.concat((enr_train[[col for col in enr_train.columns if col in ["enrollment_id",'username','course_id'] ]], enr_test))
count_lk = {col: X.groupby(col).aggregate({"enrollment_id": "count"}).to_dict()["enrollment_id"] for col in X.columns}

for col in ['username','course_id']:
    enr_train[col + "_count"] = ([count_lk[col][x] for x in enr_train[col]])
    enr_test[col + "_count"] = ([count_lk[col][x] for x in enr_test[col]])


#enr_trainにlog_trainのデータを付加する
for col in ['source','event','year','month','day','hour','weekday','module_depth','category']:#'source',
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

#log_trainでの各idの連続する数を追加する
enr_train['sum_count'] = enr_train[['server','browser']].sum(1)
enr_train = enr_train.fillna(0)

#del train['0_cg']

#enr_testにlog_testのデータを付加する
for col in ['source','event','year','month','day','hour','weekday','module_depth','category']:
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



#enr_trainvalを1:2にsplit
n_val = round(len(enr_train)*0.2)
np.random.seed(19920407)
rows = np.random.choice(enr_train.index.values, len(enr_train),replace=False)

new_train = enr_train.ix[rows[n_val:]]
val_data = enr_train.ix[rows[:n_val]]
#shuffle_train = enr_train.ix[rows]
shuffle_train = enr_train

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
val_label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/val_enr_label.csv',index=False)






#log transform
f = lambda x: np.log(1+x**2) if x > 1 else x



#for i in xrange(0,100):
#    print i,ggg.iloc[(ggg['obj_count']==i).values,:].drop_duplicates('enrollment_id')['drop'].value_counts()

dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train_depth.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test_depth.csv')
label = dtrain['drop']
'''
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train_depth.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test_depth.csv')
X = pd.concat([train,test],axis=0,ignore_index=True)
obj_count = X['object'].value_counts()
obj_count = pd.DataFrame(obj_count,columns=['obj_count'])
obj_count['object'] = obj_count.index
gg = X.merge(obj_count,on='object',how='left').sort('enrollment_id')
label_ = dtrain[['enrollment_id','drop']]
ggg = gg.merge(label_,on='enrollment_id',how='left').sort('enrollment_id')
#log_trainのobjectのカウントを取りidごとの最小の値を追加
ggg2 = ggg.groupby('enrollment_id').min()['obj_count']
ggg2.index = range(0,len(ggg2))
dtrain['obj_count_min'] = ggg2[:len(dtrain)]
dtest['obj_count_min'] = ggg2[len(dtrain):].values
'''
#ggg2 = ggg.groupby('enrollment_id').max()['obj_count']
#ggg2.index = range(0,len(ggg2))
#dtrain['obj_count_max'] = ggg2[:len(dtrain)]
#dtest['obj_count_max'] = ggg2[len(dtrain):].values
#ggg.iloc[ggg['module_depth'].notnull().values,:]
#ggg2 = ggg.groupby('enrollment_id').mean()['obj_count']
#ggg2.index = range(0,len(ggg2))
#dtrain['obj_count_mean'] = ggg2[:len(dtrain)]
#dtest['obj_count_mean'] = ggg2[len(dtrain):].values




del dtrain['drop']

X = pd.concat([dtrain,dtest])
X.index = range(0,len(X))

#course_idのダミー変数化
b = categorical(np.array(X['course_id']), drop=True)
b = pd.DataFrame(b)

#aaa = pd.factorize(X['username'])#usernameのFactorize化
#dtrain['username'] = aaa[0][:len(dtrain)]
#dtrain[dtrain['username_count'] <= 5]['username'] = -999


#bbb = pd.factorize(X['course_id'])#course_idのFactorize化
#dtrain['course_id'] = bbb[0][:len(dtrain)]

del dtrain['enrollment_id'],dtrain['username'],dtrain['course_id'],
dtrain = pd.concat([b[:len(dtrain)],dtrain],axis=1)
#label = dtrain['drop']
#del dtrain['drop']
#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換


#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv')
#b = categorical(np.array(dtest['course_id']), drop=True)
#b = pd.DataFrame(b)
#dtest['username'] = aaa[0][len(dtrain):]

#dtest['course_id'] = bbb[0][len(dtrain):]



del dtest['enrollment_id'],dtest['username'], dtest['course_id']

xx = b[len(dtrain):]
xx.index = range(0,len(xx))
dtest = pd.concat([xx,dtest],axis=1)
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換

#year, month, day, hour ,weekdayのカウントデータを1にする
f_to1 = lambda x: 1 if x >=1 else 0
label_to1 = [u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w']

dtrain[label_to1] = dtrain[label_to1].applymap(f_to1)
dtest[label_to1] = dtest[label_to1].applymap(f_to1)

label_to1 = [u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki']
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

#enrollment_trainとenrollment_idを合わせるためにソート
train = train.sort('enrollment_id')
test = test.sort('enrollment_id')

f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
train_id_time = train[['enrollment_id','time']]
test_id_time = test[['enrollment_id','time']]
train_id_time['time'] = train_id_time['time'].apply(f_time)
test_id_time['time'] = test_id_time['time'].apply(f_time)

a = np.zeros(len(dtrain)*4)
a = a.reshape((len(dtrain),4))
train_time = pd.DataFrame(a,columns=['min_time','max_time','mean_time','len_time'])
train_time.columns = ['min_time','max_time','mean_time','len_time']
i = 0
a = train_id_time['enrollment_id'].drop_duplicates().values
#f_toordinal = lambda x: x.toordinal() - 735000
for id in a:
    train_time_ = train_id_time.iloc[(train_id_time['enrollment_id']==id).values,1]
    train_time.loc[i,['min_time']] = min(train_time_)
    train_time.loc[i,['max_time']] = max(train_time_)
    #train_id_time['time'] = train_id_time['time'].apply(f_toordinal)
    train_time.loc[i,['mean_time']] = np.mean(train_time_.values)
    i+=1
    if i % 1000 == 0:
        print i
train_time['len_time'] = train_time['max_time'] - train_time['min_time']

test_time = pd.DataFrame(np.zeros(len(dtest)),columns=['time'])
train_time.columns = ['min_time','max_time','mean_time','len_time']

a = np.zeros(len(dtest)*4)
a = a.reshape((len(dtest),4))
test_time = pd.DataFrame(a,columns=['min_time','max_time','mean_time','len_time'])
test_time.columns = ['min_time','max_time','mean_time','len_time']
i = 0
a = test_id_time['enrollment_id'].drop_duplicates().values
for id in a:
    test_time_ = test_id_time.iloc[(test_id_time['enrollment_id']==id).values,1]
    test_time.loc[i,['min_time']] = min(test_time_)
    test_time.loc[i,['max_time']] = max(test_time_)

    #test_id_time['time'] = test_id_time['time'].apply(f_toordinal)
    test_time.loc[i,['mean_time']] = np.mean(test_time_.values)

    i+=1
    if i % 1000 == 0:
        print i
test_time['len_time'] = test_time['max_time'] - test_time['min_time']
['min_time','max_time','mean_time','len_time']

dtrain[['min_time','max_time','mean_time','len_time']] = train_time
dtest[['min_time','max_time','mean_time','len_time']] = test_time

#train_timelenを読み込んでdtrainに付け加える
train_time.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen_mean.csv',index=False)
test_time.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen_mean.csv',index=False)
'''

#shuffleしてないenr_trainに追加しろ!!
dtrain[['min_time','max_time','mean_time','len_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen_mean.csv')
dtest[['min_time','max_time','mean_time','len_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen_mean.csv')

dtrain['max-mean']=dtrain['max_time']-dtrain['mean_time']
dtrain['mean-min']=dtrain['mean_time']-dtrain['min_time']

dtest['max-mean']=dtest['max_time']-dtest['mean_time']
dtest['mean-min']=dtest['mean_time']-dtest['min_time']

'''
log_trainのなかのmodule_idをstartとの差で表す
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')

Object = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/object.csv')
Object = Object.fillna('0')
Object = Object.drop_duplicates()
Object.index = range(0,len(Object))

train = train.merge(Object[['module_id','start']],how='left',left_on='object',right_on='module_id')
del train['module_id']
train['start'] = train['start'].fillna('null')
f_toordinal = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000 if x != 'null' else 100000

train['time'] = train['time'].apply(f_toordinal)
test['time'] = test['time'].apply(f_toordinal)

train['start'] = train['start'].apply(f_toordinal)
train['diff_time'] = train['time'] - train['start']
f_ela = lambda x: 0 if x < -90000 else x
train['diff_time'] = train['diff_time'].apply(f_ela)

ddd = train.groupby('enrollment_id')[['start','diff_time']].mean()
ddd.index = range(0,len(ddd))
dtrain[['start','diff_time']] = ddd
#train_course = dtrain['course_id'] +1
#test_course = dtest['course_id'] +1
'''

#dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')

#dtrain = dtrain.iloc[:,39:]
#dtest = dtest.iloc[:,39:]

f_to0 = lambda x: -1 if x == 0 else x
#dtrain.iloc[:,:39] = dtrain.iloc[:,:39].applymap(f_to0)
#dtest.iloc[:,:39] = dtest.iloc[:,:39].applymap(f_to0)
#dtrain['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
#dtest['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')



'''
dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_NN.csv',index=False)
dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/label_NN.csv',index=False)
'''
aa = dtrain.columns

#del dtrain['object_fact'], dtest['object_fact']

#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換
'''
numerics = dtrain.loc[:, [u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki',u'sum_count', u'min_time', u'max_time', u'mean_time', u'len_time', u'max-mean', u'mean-min']]
 
# for each pair of variables, determine which mathmatical operators to use based on redundancy
for i in range(0, numerics.columns.size-1):
    for j in range(0, numerics.columns.size-1):
        col1 = str(numerics.columns.values[i])
        col2 = str(numerics.columns.values[j])
        # multiply fields together (we allow values to be squared)
        if i <= j:
            name = col1 + "*" + col2
            dtrain = pd.concat([dtrain, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
        # add fields together
        #if i < j:
            #name = col1 + "+" + col2
            #dtrain = pd.concat([dtrain, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis=1)
        # divide and subtract fields from each other
        #if not i == j:
            #name = col1 + "/" + col2
            #dtrain = pd.concat([dtrain, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name=name)], axis=1)
            #name = col1 + "-" + col2
            #dtrain = pd.concat([dtrain, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)

#numerics = dtest.loc[:, [u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki']]
numerics = dtest.loc[:, [u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki',u'sum_count', u'min_time', u'max_time', u'mean_time', u'len_time', u'max-mean', u'mean-min']]
# for each pair of variables, determine which mathmatical operators to use based on redundancy
for i in range(0, numerics.columns.size-1):
    for j in range(0, numerics.columns.size-1):
        col1 = str(numerics.columns.values[i])
        col2 = str(numerics.columns.values[j])
        # multiply fields together (we allow values to be squared)
        if i <= j:
            name = col1 + "*" + col2
            dtest = pd.concat([dtest, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
        # add fields together
        #if i < j:
            #name = col1 + "+" + col2
            #dtest = pd.concat([dtest, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis=1)
        # divide and subtract fields from each other
        #if not i == j:
            #name = col1 + "/" + col2
            #dtest = pd.concat([dtest, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name=name)], axis=1)
            #name = col1 + "-" + col2
            #dtest = pd.concat([dtest, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)
'''









dtrain_leaf = dtrain
dtest_leaf = dtest
'''
del dtrain_leaf['start']
#2つのやまがあるのでそれぞれ分けて学習してみる
dtrain_leaf.iloc[(dtrain_leaf.loc[:,['max_time']]<350).values),:]
dtest_leaf.iloc[(dtest_leaf.loc[:,['max_time']]<350).values),:]
'''
#dtrain_leaf['course_id'] = bbb[0][:len(dtrain)]
#dtest_leaf['course_id'] = bbb[0][len(dtrain):]
#ddd = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'sum_count', u'min_time', u'max_time', u'len_time',u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m']


#dtrain_leaf.loc[(dtrain_leaf['min_time']>350).values,'min_time'] = dtrain_leaf.loc[(dtrain_leaf['min_time']>350).values,'min_time'] - 198
#dtrain_leaf.loc[(dtrain_leaf['max_time']>350).values,'max_time'] = dtrain_leaf.loc[(dtrain_leaf['max_time']>350).values,'max_time'] - 198
#dtrain_leaf.loc[(dtrain_leaf['mean_time']>350).values,'mean_time'] = dtrain_leaf.loc[(dtrain_leaf['mean_time']>350).values,'mean_time'] - 198
'''
dtrain['obj_not_nan'] = dtrain[[u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md']].sum(1)
dtest['obj_not_nan'] = dtest[[u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md']].sum(1)
dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/NN/train_NN.csv',index=False)
dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/label_NN.csv',index=False)


'''
using_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h',u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'0.0_md', u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md', u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg', u'sum_count', u'obj_count_min', u'min_time', u'max_time', u'mean_time', u'len_time', u'max-mean', u'mean-min']

using_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h',u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'0.0_md', u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md', u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg', u'sum_count', u'obj_count_min', u'min_time', u'max_time', u'mean_time', u'len_time', u'max-mean', u'mean-min']

dtrain = xgb.DMatrix(dtrain_leaf[using_label], label=label)
dtest = xgb.DMatrix(dtest_leaf)


#366-168=198

f_toNaN





dtrain_leaf['drop'] = label
xxx1 = dtrain_leaf.iloc[(dtrain_leaf['max_time']>325).values,:]
label_xxx1 = xxx1['drop']
xxx2 = dtrain_leaf.iloc[(dtrain_leaf['max_time']<325).values,:]
label_xxx2 = xxx2['drop']
preds_tr = xxx2
preds_tr.index = range(0,len(xxx2))
preds_te = dtest_leaf

del xxx1['drop'],xxx2['drop']

dtrain1 = xgb.DMatrix(xxx1,label=label_xxx1)
dtrain2 = xgb.DMatrix(xxx2,label=label_xxx1) 

#dtest_leaf.iloc[(dtest_leaf.loc[:,['max_time']]<350).values),:]

pred_name = []
num_model = 30
for i in xrange(0,num_model):
    random.seed(i*1302352)
    

    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(5,5),
              "bst:eta" :  random.uniform(.001, .01),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,4),
              "bst:subsample" :  round(random.uniform(.1, 1.),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.1, 1.),2),#column_subsample
              "silent": 1,
              "base_score": 0.55,
              #"max_delta_step": 3,
              #"scale_pos_weight": 0.95,
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
    watchlist  = [(dtrain,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,watchlist)

    bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    bst = pd.Series(bst).apply(f_)
    num_round_ = bst.argmax() + 35
    cc = bst.max()
    if cc > 0.8863:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,len(pred_name)):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(len(pred_name))
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred45.csv',header=None,index=False)













preds_tr = pd.concat([preds_tr,pd.DataFrame(pred_prob1)],ignore_index=True,axis=1)
        preds_te = pd.concat([preds_te,pd.DataFrame(pred_prob2)],ignore_index=True,axis=1)

preds_tr.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/train_NN.csv',index=False)
preds_te.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/test_NN.csv',index=False)
label_xxx2.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/label_NN.csv',index=False)

label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/label_NN.csv',header=None)

def csv_to_vw(loc_csv, loc_output, train=True,labels=pd.DataFrame()):
  """
  Munges a CSV file (loc_csv) to a VW file (loc_output). Set "train"
  to False when munging a test set.
  TODO: Too slow for a daily cron job. Try optimize, Pandas or Go.
  """
  start = datetime.now()
  print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))
  
  with open(loc_output,"wb") as outfile:
    for e, row in enumerate( DictReader(open(loc_csv)) ):
	
	  #Creating the features
      numerical_features = ""
      categorical_features = ""
      for k,v in row.items():
        if k not in ['enrollment_id','drop']:
          if k not in [u'username', u'course_id']: # numerical feature, example: I5
            if len(str(v)) > 0: #check for empty values
              numerical_features += " %s:%s" % (k,v)
          if k in [u'username', u'course_id']: # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v
			  
	  #Creating the labels		  
      if train: #we care about labels
        if row['drop'] == '1':
          label_ = 1
        else:
          label_ = -1 #we set negative label to -1
        outfile.write( "%s |i%s |c%s\n" % (label_, numerical_features, categorical_features) )
		
      else: #we dont care about labels
        outfile.write( "1 |i%s |c%s\n" % (numerical_features, categorical_features) )
      
	  #Reporting progress
      if e % 10000 == 0:
        print("%s\t%s"%(e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))

#trainをvalidationdataとにsplit
dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/train_NN.csv')
#del dtrain['drop']
dtrain['drop'] = label

#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_NN.csv',header=None)
dtrain.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/shuffle_enr_train2.csv',index=False)

n_val = round(len(dtrain)*0.2)
rows = np.random.choice(dtrain.index.values, len(dtrain),replace=False)

new_train = dtrain.ix[rows[n_val:]]
val = dtrain.ix[rows[:n_val]]

new_train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/new_train.csv',index=False)
val.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/new_val.csv',index=False)
val_label = val['drop']
val_label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/val_enr_label2.csv',index=False)

#train, validation
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/new_train.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/new_enr_train2.vw',train=True,labels=label)
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/new_val.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/enr_validation2.vw',train=True,labels=label)

#shuffle train
#csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/shuffle_enr_train2.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/shuffle_enr_train2.vw',train=True,labels=label)

#test
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/test_NN.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/new_enr_test2.vw',train=False)



#学習
#decayも変更せよ
!vw fe_train/fe_vw/new_enr_train2.vw -k -c -f fe_train/fe_vw/train.model2.vw --loss_function logistic --passes 15 -l 0.05 -b 30 --holdout_off --random_seed 19920407 -q ii

#!vw fe_train/fe_vw/new_enr_train2.vw -k -c -f fe_train/fe_vw/train.model2.vw --loss_function quantile --passes 1 -l 0.6 -b 30 --holdout_off --cubic ccc --quantile_tau 0.38 --noconstant --random_seed 19920407

!vw fe_train/fe_vw/enr_validation2.vw -t -i fe_train/fe_vw/train.model2.vw -p fe_test/fe_vw/test2_val.txt --invert_hash model_readable.txt



#!vw fe_train/fe_vw/enr_validation2.vw -t -i fe_train/fe_vw/train.model2.vw --loss_function hinge --passes 1 -l 5.0 -b 30 --holdout_off --cubic ccc


outputfile = "/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/pred2_val.csv" 

def zygmoid(x):
    return 1 / (1 + math.exp(-x))

with open(outputfile,"wb") as outfile:
    #outfile.write("Id,Predicted\n")
    for line in open("fe_test/fe_vw/test2_val.txt"):
        row = line.strip().split(" ")
        outfile.write("%f\n"%(zygmoid(float(row[0]))))

#ROC
pred_val = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/pred2_val.csv',header=None)

y = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/val_enr_label2.csv',header=None)
print pd.concat([y[0],pred_val[0]],axis=1)
#pred_val[0][pred_val[0]<0.1] = 0.001
#pred_val[0][pred_val[0]>0.96] = 0.999
fpr, tpr, thresholds = roc_curve(y, pd.DataFrame(list(pred_val[0])), pos_label=1)  
print auc(fpr, tpr)
plt.plot(fpr,tpr)
plt.show()

#テスト
!vw fe_test/fe_vw/new_enr_test2.vw -t -i fe_train/fe_vw/train.model2.vw -p fe_test/fe_vw/test2.txt

outputfile = "/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/pred.csv" 
with open(outputfile,"wb") as outfile:
    #outfile.write("Id,Predicted\n")
    for line in open("fe_test/fe_vw/test2.txt"):
        row = line.strip().split(" ")
        outfile.write("%f\n"%(zygmoid(float(row[0]))))

pred = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/pred.csv',header=None)
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)

Pred = pd.concat([sample[0],pred[0]],axis=1)

Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/Pred_vw1.csv',header=None,index=False)


