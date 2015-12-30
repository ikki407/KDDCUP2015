# -*- coding: UTF-8 -*-

###################################
#feature engineering
#xgboost
#5-foldCV
###################################

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

#曜日をつける
f_w = lambda x: str(date(int(x[:4]),int(x[5:7]),int(x[8:10])).weekday())
train['weekday'] = train['time'].apply(f_w)
test['weekday'] = test['time'].apply(f_w)

del train['time']
del test['time']

#LibFFM用
#n_val = round(len(train)*0.2)
#rows = np.random.choice(train.index.values, len(train))
#train = train.ix[rows]
#train.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_fm.csv', index=False)
#test.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv', index=False)


#enr_trainの中のusernameとcourse_idのカウント数を各idに追加する
X = pd.concat((enr_train[[col for col in enr_train.columns if col in ["enrollment_id",'username','course_id'] ]], enr_test))
count_lk = {col: X.groupby(col).aggregate({"enrollment_id": "count"}).to_dict()["enrollment_id"] for col in X.columns}

for col in ['username','course_id']:
    enr_train[col + "_count"] = ([count_lk[col][x] for x in enr_train[col]])
    enr_test[col + "_count"] = ([count_lk[col][x] for x in enr_test[col]])


#enr_trainにlog_trainのデータを付加する
for col in ['source','event','year','month','day','hour','weekday']:
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

    enr_train = pd.concat([enr_train,col_count],axis=1)

#log_trainでの各idの連続する数を追加する
enr_train['sum_count'] = enr_train[['server','browser']].sum(1)
enr_train = enr_train.fillna(0)


#enr_testにlog_testのデータを付加する
for col in ['source','event','year','month','day','hour','weekday']:
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
#Object = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/object.csv')
#Object['module_id'].columns = ['object']
#Object.merge(enr_train, on='object', how='inner')[['category','children']]



#enr_trainvalを1:2にsplit
n_val = round(len(enr_train)*0.2)
np.random.seed(19920407)
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






#log transform
f = lambda x: np.log(1+x) if x > 1 else x

dtrain_sp = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/new_enr_train.csv')
dval = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/enr_validation.csv')

label_dtrain_sp = dtrain_sp['drop']
del dtrain_sp['drop']
label_dval = dval['drop']
del dval['drop']

#course_idのダミー変数化
X = pd.concat([dtrain_sp,dval])
X.index = range(0,len(X))
#b = categorical(np.array(X['course_id']), drop=True)
#b = pd.DataFrame(b)

aaa = pd.factorize(X['username'])#usernameのFactorize化
dtrain_sp['username'] = aaa[0][:len(dtrain_sp)]

bbb = pd.factorize(X['course_id'])#course_idのFactorize化
dtrain_sp['course_id'] = bbb[0][:len(dtrain_sp)]


del dtrain_sp['enrollment_id'], dtrain_sp['course_id'],dtrain_sp['username']
#dtrain_sp = pd.concat([b[:len(dtrain_sp)],dtrain_sp],axis=1)
#label_dtrain_sp = dtrain_sp['drop']
#del dtrain_sp['drop']
#w = np.random.rand(len(dtrain_sp),1)
#dtrain_sp.iloc[:,39:] = dtrain_sp.iloc[:,39:].applymap(f)#対数変換
aa = dtrain_sp.columns


#dval = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/enr_validation.csv')
#b = categorical(np.array(dval['course_id']), drop=True)
#b = pd.DataFrame(b)
dval['username'] = aaa[0][len(dtrain_sp):]

del dval['enrollment_id'], dval['course_id']
dval = pd.concat([b[len(dtrain_sp):],dval],axis=1)
#label_dval = dval['drop']
#del dval['drop']
#w = np.random.rand(len(dval),1)
#dval.iloc[:,39:] = dval.iloc[:,39:].applymap(f)#対数変換

dtrain_sp = xgb.DMatrix(dtrain_sp, label=label_dtrain_sp)
dval = xgb.DMatrix(dval, label=label_dval)



dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv')

label = dtrain['drop']
del dtrain['drop']

X = pd.concat([dtrain,dtest])
X.index = range(0,len(X))

#course_idのダミー変数化
#b = categorical(np.array(X['course_id']), drop=True)
#b = pd.DataFrame(b)

#aaa = pd.factorize(X['username'])#usernameのFactorize化
#dtrain['username'] = aaa[0][:len(dtrain)]
#dtrain[dtrain['username_count'] <= 5]['username'] = -999


#bbb = pd.factorize(X['course_id'])#course_idのFactorize化
#dtrain['course_id'] = bbb[0][:len(dtrain)]

del dtrain['course_id'],dtrain['enrollment_id'],dtrain['username']
#dtrain = pd.concat([b[:len(dtrain)],dtrain],axis=1)
#label = dtrain['drop']
#del dtrain['drop']
#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換


#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv')
#b = categorical(np.array(dtest['course_id']), drop=True)
#b = pd.DataFrame(b)
#dtest['username'] = aaa[0][len(dtrain):]

#dtest['course_id'] = bbb[0][len(dtrain):]



del dtest['enrollment_id'], dtest['course_id'],dtest['username']

#xx = b[len(dtrain):]
#xx.index = range(0,len(xx))
#dtest = pd.concat([xx,dtest],axis=1)
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換

#dtrain.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_NN.csv',index=False)
#dtest.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_NN.csv',index=False)
aa = dtrain.columns
#year, month, day, hour ,weekdayのカウントデータを1にする
f_to1 = lambda x: 1 if x >=1 else 0
label_to1 = [u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w']

dtrain[label_to1] = dtrain[label_to1].applymap(f_to1)
dtest[label_to1] = dtest[label_to1].applymap(f_to1)

dtrain_leaf = dtrain
dtest_leaf = dtest

dtrain = xgb.DMatrix(dtrain, label=label)
dtest = xgb.DMatrix(dtest)




def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 1)) / np.sum(label==0)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

#0.8816
pred_name = []
num_model = 50
for i in xrange(0,num_model):
    random.seed(i)

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  round(random.uniform(.1, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(300, 400),0))
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
    #num_round_ = bst.argmax()
    #cc = bst.max()
    if True:
        watchlist  = [(dtrain,'train')]
        random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]))
        print preds
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred13_nocourseid.csv',header=None,index=False)

#葉のindexをつけて予測する
pred_name = []
num_model = 50
for i in xrange(0,num_model):
    random.seed(i)

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  round(random.uniform(.5, .8),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    num_round = 30 #int( round(random.uniform(200, 300),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    watchlist  = [(dtrain,'train')]
    bst = xgb.train(param,dtrain, num_round,watchlist)
    #leaf index
    pred_leaf_tr = bst.predict( dtrain ,pred_leaf = True)
    pred_leaf_te = bst.predict( dtest ,pred_leaf = True)
    
    dtrain_leaf2 = pd.concat([dtrain_leaf, pd.DataFrame(pred_leaf_tr)],axis=1)
    dtest_leaf2 = pd.concat([dtest_leaf, pd.DataFrame(pred_leaf_te)],axis=1)

    dtrain_leaf2 = xgb.DMatrix(dtrain_leaf2, label=label)
    dtest_leaf2 = xgb.DMatrix(dtest_leaf2)
    
    random.seed(i*999)
    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  round(random.uniform(.1, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(300, 400),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #watchlist  = [(dtrain_leaf,'train')]
    #bst = xgb.train(param,dtrain_leaf, num_round,watchlist)

    #bst=xgb.cv(param, dtrain_leaf, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    #num_round_ = bst.argmax()
    #cc = bst.max()
    if True:
        watchlist  = [(dtrain_leaf2,'train')]
        random.seed(19920407)
        bst = xgb.train(param,dtrain_leaf2, num_round,watchlist)
        pred_prob = bst.predict( dtest_leaf2 )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]))
        print preds
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred14_nocourseid.csv',header=None,index=False)

label_spw = dtrain.get_label()
scale_pos_weight = float(np.sum(label_spw == 0)) / np.sum(label_spw==1)

pred_name = []
num_model = 50
for i in xrange(0,num_model):
    random.seed(i)

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  round(random.uniform(.5, .8),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              "scale_pos_weight": scale_pos_weight,
              }
    num_round = 30 #int( round(random.uniform(200, 300),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    watchlist  = [(dtrain,'train')]
    bst = xgb.train(param,dtrain, num_round,watchlist)
    #leaf index
    pred_leaf_tr = bst.predict( dtrain ,pred_leaf = True)
    pred_leaf_te = bst.predict( dtest ,pred_leaf = True)
    
    dtrain_leaf2 = pd.concat([dtrain_leaf, pd.DataFrame(pred_leaf_tr)],axis=1)
    dtest_leaf2 = pd.concat([dtest_leaf, pd.DataFrame(pred_leaf_te)],axis=1)

    dtrain_leaf2 = xgb.DMatrix(dtrain_leaf2, label=label)
    dtest_leaf2 = xgb.DMatrix(dtest_leaf2)
    
    random.seed(i*999)
    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(300, 400),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #watchlist  = [(dtrain_leaf,'train')]
    #bst = xgb.train(param,dtrain_leaf, num_round,watchlist)

    #bst=xgb.cv(param, dtrain_leaf, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    #num_round_ = bst.argmax()
    #cc = bst.max()
    if True:
        watchlist  = [(dtrain_leaf2,'train')]
        random.seed(19920407)
        bst = xgb.train(param,dtrain_leaf2, num_round,watchlist)
        pred_prob = bst.predict( dtest_leaf2 )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]))
        print preds
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred15_nocourseid.csv',header=None,index=False)


pred_name = []
num_model = 50
for i in xrange(0,num_model):
    random.seed(i)

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  round(random.uniform(.5, .8),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.2, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, .8),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    num_round = 50 #int( round(random.uniform(200, 300),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    watchlist  = [(dtrain,'train')]
    bst = xgb.train(param,dtrain, num_round,watchlist)
    #leaf index
    pred_leaf_tr = bst.predict( dtrain ,pred_leaf = True)
    pred_leaf_te = bst.predict( dtest ,pred_leaf = True)
    
    dtrain_leaf2 = pd.concat([dtrain_leaf, pd.DataFrame(pred_leaf_tr)],axis=1)
    dtest_leaf2 = pd.concat([dtest_leaf, pd.DataFrame(pred_leaf_te)],axis=1)

    dtrain_leaf2 = xgb.DMatrix(dtrain_leaf2, label=label)
    dtest_leaf2 = xgb.DMatrix(dtest_leaf2)
    
    random.seed(i*999)
    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  round(random.uniform(.1, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.2, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, .9),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(300, 400),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain_sp,'train')]
    #bst = xgb.train(param,dtrain_sp, num_round,evallist)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #watchlist  = [(dtrain_leaf,'train')]
    #bst = xgb.train(param,dtrain_leaf, num_round,watchlist)

    #bst=xgb.cv(param, dtrain_leaf, num_round, nfold=5, seed = 19920407, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    #num_round_ = bst.argmax()
    #cc = bst.max()
    if True:
        watchlist  = [(dtrain_leaf2,'train')]
        random.seed(19920407)
        bst = xgb.train(param,dtrain_leaf2, num_round,watchlist)
        pred_prob = bst.predict( dtest_leaf2 )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]))
        print preds
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred16_nocourseid.csv',header=None,index=False)


