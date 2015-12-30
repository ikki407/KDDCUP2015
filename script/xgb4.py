# -*- coding: UTF-8 -*-

###################################
#feature engineering
#xgboost
#2:8split cross-validation
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




dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv')

label = dtrain['drop']
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

del dtrain['course_id'],dtrain['enrollment_id'],dtrain['username']
dtrain = pd.concat([b[:len(dtrain)],dtrain],axis=1)
#label = dtrain['drop']
#del dtrain['drop']
#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換


#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv')
#b = categorical(np.array(dtest['course_id']), drop=True)
#b = pd.DataFrame(b)
#dtest['username'] = aaa[0][len(dtrain):]

#dtest['course_id'] = bbb[0][len(dtrain):]



del dtest['enrollment_id'], dtest['course_id'],dtest['username']

xx = b[len(dtrain):]
xx.index = range(0,len(xx))
dtest = pd.concat([xx,dtest],axis=1)
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換

#dtrain.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_NN.csv',index=False)
#dtest.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_NN.csv',index=False)
aa = dtrain.columns
#year, month, day, hour ,weekdayのカウントデータを1にする
f_to1 = lambda x: 1 if x >=1 else 0
label_to1 = [u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w']

dtrain[label_to1] = dtrain[label_to1].applymap(f_to1)
dtest[label_to1] = dtest[label_to1].applymap(f_to1)


#enr_trainvalを1:2にsplit
n_val = round(len(dtrain)*0.2)
np.random.seed(19920407)
rows = np.random.choice(dtrain.index.values, len(dtrain))
dtrain_sp = dtrain.ix[rows[n_val:]]
dval = dtrain.ix[rows[:n_val]]

np.random.seed(19920407)
rows = np.random.choice(dtrain.index.values, len(dtrain))
label_dtrain = label.ix[rows[n_val:]]
label_dval = label.ix[rows[:n_val]]

print len(dtrain_sp)==len(label_dtrain), len(dval)==len(label_dval)
dtrain = xgb.DMatrix(dtrain_sp, label=label_dtrain)
dval = xgb.DMatrix(dval, label=label_dval)
dtest = xgb.DMatrix(dtest)




def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 1)) / np.sum(label==0)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

label_spw = dtrain.get_label()
scale_pos_weight = float(np.sum(label_spw == 0)) / np.sum(label_spw==1)
pred_name = []
for i in xrange(0,50):
    random.seed(i)
    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  round(random.uniform(.1, .3),2),#step_size
              "bst:gamma" :  random.randint(0,2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.2, .4),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.6, 1.0),2),#column_subsample
              "silent": 1,
              "scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(300, 400),0))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    evallist  = [(dval, 'eval'),(dtrain,'train')]
    bst = xgb.train(param,dtrain, num_round,evallist)
    pred_prob = bst.predict( dtest )
    print pd.DataFrame(pred_prob)
    #watchlist  = [(dtrain,'train')]
    #bst = xgb.train(param,dtrain, num_round,watchlist)

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 19920407, show_stdv = False, fpreproc = fpreproc)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    
    #pred_prob = bst.predict( dval )
    #print pred_prob
    #ROC
    #pred_val = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/pred_val.csv',header=None)
    #y = dval.get_label()
    #fpr, tpr, thresholds = roc_curve(y, pred_prob, pos_label=1)  
    #print auc(fpr, tpr)
    #plt.plot(fpr,tpr)
    #plt.show()





    f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    bst = pd.Series(bst).apply(f_)
    num_round_ = bst.argmax()
    cc = bst.max()
    if False:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)

#Pred = pd.DataFrame()
#for i in xrange(0,50):
#    pred = pd.read_csv(pred_name[i], header=None)
#    Pred = Pred + pred[1]
#Pred = Pred/50.0
#Pred = pd.concat([pred[0],Pred],axis=1)
#Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred1.csv',header=None,index=False)

#param = {'bst:subsample': 0.87, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.56, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 12, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
#num_round = 400
#watchlist  = [(dtrain_all_sf,'train')]
#bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
#print param


#予測
#0.947205
param = {'bst:max_depth': 22, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.59, 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.18, 'bst:min_child_weight': 1}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred1_all.csv',header=None,index=False)

#0.948055
param = {'bst:max_depth': 20, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.93, 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.16, 'bst:min_child_weight': 3}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred2_all.csv',header=None,index=False)

#0.943450
param = {'bst:max_depth': 13, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.66, 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.15, 'bst:min_child_weight': 3}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred3_all.csv',header=None,index=False)

#0.944048
param = {'bst:max_depth': 19, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.7, 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.23, 'bst:min_child_weight': 3}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred4_all.csv',header=None,index=False)

#0.943016
param = {'bst:max_depth': 22, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.91, 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.24, 'bst:min_child_weight': 7}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred5_all.csv',header=None,index=False)

#0.943016
param = {'bst:max_depth': 18, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.26, 'bst:min_child_weight': 2}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred6_all.csv',header=None,index=False)

#0.943016
param = {'bst:max_depth': 15, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.12, 'bst:min_child_weight': 3}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred7_all.csv',header=None,index=False)

#0.9514
param = {'bst:subsample': 0.51, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.06, 'colsample_bytree': 0.96, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 15, 'objective': 'binary:logistic', 'bst:min_child_weight': 0}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred8_all.csv',header=None,index=False)






pred1 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred1_all.csv',header=None)
pred2 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred2_all.csv',header=None)
pred3 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred3_all.csv',header=None)
pred4 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred4_all.csv',header=None)
pred5 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred5_all.csv',header=None)
pred6 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred6_all.csv',header=None)
pred7 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred7_all.csv',header=None)
pred8 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred8_all.csv',header=None)

final_pred = (pred1[1]+pred2[1]+pred3[1]+pred4[1]+pred5[1]+pred6[1]+pred7[1]+pred8[1])/8.0
final_pred = pd.concat([pred1[0],final_pred],axis=1)
final_pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ens2_all.csv',header=None,index=False)


#0.940756
param = {'bst:subsample': 0.84, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.26, 'colsample_bytree': 0.85, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 28, 'objective': 'binary:logistic', 'bst:min_child_weight': 0}
num_round = 240
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred8_all.csv',header=None,index=False)


pred = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred8_all.csv',header=None)



'''
0.8697894229106189
prediction1_all
{'bst:max_depth': 17, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.07, 'bst:min_child_weight': 3}
78
'''

#feature_importance
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





#0.8739567593608065
param = {'bst:subsample': 0.51, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.06, 'colsample_bytree': 0.96, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 15, 'objective': 'binary:logistic', 'bst:min_child_weight': 0}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred9_all.csv',header=None,index=False)

#0.872988969666523
param = {'bst:subsample': 0.51, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.06, 'colsample_bytree': 0.96, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 15, 'objective': 'binary:logistic', 'bst:min_child_weight': 0}
num_round = 150
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred10_all.csv',header=None,index=False)

#0.8704286487301565
param = {'bst:subsample': 0.51, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.06, 'colsample_bytree': 0.96, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 15, 'objective': 'binary:logistic', 'bst:min_child_weight': 0}
num_round = 50
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred11_all.csv',header=None,index=False)

#0.8648408712838
#num_round = 250

param = {'bst:subsample': 0.67, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.19, 'colsample_bytree': 0.77, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 17, 'objective': 'binary:logistic', 'bst:min_child_weight': 0}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred12_all.csv',header=None,index=False)

param = {'bst:subsample': 0.95, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.15, 'colsample_bytree': 0.94, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 21, 'objective': 'binary:logistic', 'bst:min_child_weight': 1}
num_round = 110
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred13_all.csv',header=None,index=False)


pred1 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred12_all.csv',header=None)
pred2 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred13_all.csv',header=None)

final_pred = (pred1[1]+pred2[1])/2.0
final_pred = pd.concat([pred1[0],final_pred],axis=1)
final_pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ens3_all.csv',header=None,index=False)



param = {'bst:subsample': 0.55, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.23, 'colsample_bytree': 0.82, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 22, 'objective': 'binary:logistic', 'bst:min_child_weight': 0}
num_round = 110
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred14_all.csv',header=None,index=False)

preds[preds.iloc[:,1] < 0.01] = 0.0001
preds[preds.iloc[:,1] > 0.90] = 0.99



#0.85
param = {'bst:subsample': 0.9, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.22, 'colsample_bytree': 0.94, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 23, 'objective': 'binary:logistic', 'bst:min_child_weight': 1}
num_round = 58
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred15_all.csv',header=None,index=False)


#0.8738691075822864
param = {'bst:subsample': 0.62, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.08, 'colsample_bytree': 0.65, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 10, 'objective': 'binary:logistic', 'bst:min_child_weight': 1}
num_round = 100
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred16_all.csv',header=None,index=False)

#0.88
param = {'bst:max_depth': 4, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.85, 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.11, 'bst:min_child_weight': 5}
num_round = 150
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda1.csv',header=None,index=False)

#0.918
param = {'bst:max_depth': 9, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.78, 'bst:gamma': 0, 'nthread': 8, 'bst:eta': 0.27, 'bst:min_child_weight': 3}
num_round = 80
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda2.csv',header=None,index=False)

#0.918
#0.876 #username消し, #course_idバイナリ化
param = {'bst:max_depth': 8, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.78, 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.07, 'bst:min_child_weight': 5}
num_round = 120
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda3.csv',header=None,index=False)




#0.918
param = {'bst:max_depth': 8, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.78, 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.07, 'bst:min_child_weight': 5}
num_round = 160
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda4.csv',header=None,index=False)

#0.87726
param = {'bst:max_depth': 8, 'num_class': 1, 'silent': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'bst:subsample': 0.78, 'bst:gamma': 1, 'nthread': 8, 'bst:eta': 0.07, 'bst:min_child_weight': 5}
num_round = 120
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda5.csv',header=None,index=False)

#0.87766
param = {'bst:subsample': 0.84, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.15, 'colsample_bytree': 0.78, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 4, 'objective': 'binary:logistic', 'bst:min_child_weight': 5}
num_round = 230
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda10.csv',header=None,index=False)

#0.8771
param = {'bst:subsample': 0.84, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.15, 'colsample_bytree': 0.78, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 4, 'objective': 'binary:logistic', 'bst:min_child_weight': 5}
num_round = 180
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda11.csv',header=None,index=False)

#0.87841
param = {'bst:subsample': 0.84, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.15, 'colsample_bytree': 0.78, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 4, 'objective': 'binary:logistic', 'bst:min_child_weight': 5}
num_round = 300
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda12.csv',header=None,index=False)

#
param = {'bst:subsample': 0.84, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.15, 'colsample_bytree': 0.78, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 4, 'objective': 'binary:logistic', 'bst:min_child_weight': 5}
num_round = 400
watchlist  = [(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda17.csv',header=None,index=False)





#0.8778681088986866
pred1 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred9_all.csv',header=None)
pred2 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda3.csv',header=None)

#0.877995758339326
pred1 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred9_all.csv',header=None)
pred2 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda3.csv',header=None)
pred3 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred16_all.csv',header=None)

#0.877155396865896
pred1 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred9_all.csv',header=None)
pred2 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda3.csv',header=None)
pred3 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred16_all.csv',header=None)
pred4 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all0_0.945089_171_23.csv',header=None)



#0.8778681088986866
pred1 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred9_all.csv',header=None)
pred2 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda3.csv',header=None)
pred3 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred16_all.csv',header=None)
pred4 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda5.csv',header=None)
pred5 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ikeda10.csv',header=None)


final_pred = (pred1[1]+pred2[1]+pred3[1]+pred4[1]+pred5[1])/5.0
final_pred = pd.concat([pred1[0],final_pred],axis=1)
final_pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ens9_all.csv',header=None,index=False)

preds[preds.iloc[:,1] < 0.01] = 0.0001
preds[preds.iloc[:,1] > 0.90] = 0.99




pred = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all0_0.933065_196_8.csv',header=None)
predd = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all16_0.911487_684_4.csv',header=None)


pred1.head(10)
pred2.head(10)
pred.head(10)
predd.head(10)
final_pred.head(10)

pred1.tail(10)
pred2.tail(10)
pred.tail(10)
predd.tail(10)
final_pred.tail(10)





#曜日追加
0.8727691259427788
pred_all16_0.911487_684_4.csv
{'bst:subsample': 0.77, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.22, 'colsample_bytree': 0.93, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 4, 'objective': 'binary:logistic', 'bst:min_child_weight': 2}


0.8711669791389892
pred_all78_0.923359_906_5.csv
{'bst:subsample': 0.96, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.19, 'colsample_bytree': 0.75, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 5, 'objective': 'binary:logistic', 'bst:min_child_weight': 4}


0.8648769169856781
pred_all73_0.931300_862_6.csv
{'bst:subsample': 0.78, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.2, 'colsample_bytree': 0.61, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 6, 'objective': 'binary:logistic', 'bst:min_child_weight': 4}

#0.875
ikeda13.csv
{'bst:subsample': 0.86, 'eval_metric': 'auc', 'num_class': 1, 'bst:eta': 0.21, 'colsample_bytree': 0.89, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 4, 'objective': 'binary:logistic', 'bst:min_child_weight': 1}
num_round = 300

#0.878
ikeda12.csv

#0.877
ikeda10.csv



def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 1)) / np.sum(label==0)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

pred_name = []
for i in xrange(0,50):

    param = {"objective" : "binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(3,4),
              "bst:eta" :  round(random.uniform(.1, .3),2),#step_size
              "bst:gamma" :  random.randint(0,2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.6, 1.0),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": 10000.0,
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

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 19920407, show_stdv = False, fpreproc = fpreproc)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    #num_round_ = bst.argmax()
    #cc = bst.max()
    if True:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,50):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/50.0
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred1.csv',header=None,index=False)


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
              "bst:max_depth" :  random.randint(2,2),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              #"bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              "scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(400, 700),0))
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
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred4.csv',header=None,index=False)



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
              "bst:max_depth" :  random.randint(2,2),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              #"colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(400, 700),0))
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
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred5.csv',header=None,index=False)


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
              "bst:max_depth" :  random.randint(2,4),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              #"colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    if param["bst:max_depth"] == 2:
        num_round = int( round(random.uniform(400, 700),0))
    elif param["bst:max_depth"] == 3:
        num_round = int( round(random.uniform(300, 500),0))
    elif param["bst:max_depth"] == 4:
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
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred6.csv',header=None,index=False)


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
              "bst:max_depth" :  random.randint(2,4),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              #"colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              "scale_pos_weight": scale_pos_weight,
              }
    if param["bst:max_depth"] == 2:
        num_round = int( round(random.uniform(400, 700),0))
    elif param["bst:max_depth"] == 3:
        num_round = int( round(random.uniform(300, 500),0))
    elif param["bst:max_depth"] == 4:
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
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred7.csv',header=None,index=False)



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
              "bst:max_depth" :  random.randint(2,2),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.2, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              "scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(400, 700),0))
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
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred8.csv',header=None,index=False)



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
              "bst:max_depth" :  random.randint(2,2),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.2, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(400, 700),0))
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
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred9.csv',header=None,index=False)


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
              "bst:max_depth" :  random.randint(2,4),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.2, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              #"scale_pos_weight": scale_pos_weight,
              }
    if param["bst:max_depth"] == 2:
        num_round = int( round(random.uniform(400, 700),0))
    elif param["bst:max_depth"] == 3:
        num_round = int( round(random.uniform(300, 500),0))
    elif param["bst:max_depth"] == 4:
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
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred10.csv',header=None,index=False)


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
              "bst:max_depth" :  random.randint(2,4),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.2, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              "scale_pos_weight": scale_pos_weight,
              }
    if param["bst:max_depth"] == 2:
        num_round = int( round(random.uniform(400, 700),0))
    elif param["bst:max_depth"] == 3:
        num_round = int( round(random.uniform(300, 500),0))
    elif param["bst:max_depth"] == 4:
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
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred11.csv',header=None,index=False)



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
              "bst:max_depth" :  random.randint(2,2),
              "bst:eta" :  round(random.uniform(.05, .2),2),#step_size
              "bst:gamma" :  round(random.uniform(.0, 2.0),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0,8),
              "bst:subsample" :  round(random.uniform(.5, 1.0),2),#row_subsample
              #"colsample_bytree" :  round(random.uniform(.5, 1.0),2),#column_subsample
              "silent": 1,
              "scale_pos_weight": scale_pos_weight,
              }
    num_round = int( round(random.uniform(400, 700),0))
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
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all%i_%i_%i.csv' % (i,num_round, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,num_model):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(num_model)
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred12.csv',header=None,index=False)



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
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred13.csv',header=None,index=False)



