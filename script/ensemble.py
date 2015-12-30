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

'''
fe24_2.py
-10daysつける
'''


dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train4.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test4.csv')
dtrain = dtrain.sort('enrollment_id')
dtest = dtest.sort('enrollment_id')
label = dtrain['drop']



dtrain[['min_time','max_time','mean_time','std_time','len_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen_mean2.csv')
dtest[['min_time','max_time','mean_time','std_time','len_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen_mean2.csv')


dtrain['max-mean']=dtrain['max_time']-dtrain['mean_time']
dtrain['mean-min']=dtrain['mean_time']-dtrain['min_time']

dtest['max-mean']=dtest['max_time']-dtest['mean_time']
dtest['mean-min']=dtest['mean_time']-dtest['min_time']

#date.csv追加(fe21.py追加）
course_date = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/date.csv')
f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
course_date['from'] = course_date['from'].apply(f_time)
course_date['to'] = course_date['to'].apply(f_time)

#train,testのcourse_idでmerge
dtrain = dtrain.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)
dtest = dtest.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)

dtrain.index = range(0,len(dtrain))
dtest.index = range(0,len(dtest))

dtrain['to+1'] = dtrain['to']+1
dtrain['to+10'] = dtrain['to']+10
dtest['to+1'] = dtest['to']+1
dtest['to+10'] = dtest['to']+10


#新しくできる列
#'in10days_enr','len_sum_in10days','1_min_to+1','2_min_to+1','3_min_to+1','4_min_to+1','5_min_to+1','6_min_to+1','7_min_to+1','8_min_to+1','9_min_to+1','10_min_to+1'

#そのユーザが対象のコースの１０日間で他のコースに参加しているか否か（その回数）
using_label = ['enrollment_id','username','to+1','to+10','min_time','max_time']
X = pd.concat([dtrain[using_label],dtest[using_label]])
X.index = range(0,len(X))
aaa = X['username'].drop_duplicates().values
#保存するdf
df = pd.DataFrame(np.zeros(len(X)*2).reshape(2,len(X)).T,columns=['enrollment_id','in10days_max-to+1'])
index = 0
for i in aaa:#各usernameごとにfor
    ee = X.iloc[(X['username']==i).values,:]
    if len(ee)>1:
        for j in range(0,len(ee)):
            ee2 = ee.iloc[j,:]#対象とするコースに参加しているデータ
            aa = range(0,len(ee))
            aa.remove(j)
            count4 = 0
            for k in aa:
                ee3 = ee.iloc[k,:]#比較するデータ
                if (ee2['max_time'] <= ee3['max_time'] < ee2['to+1']):
                    count4 += 1

            df.iloc[index,:] = [ee2['enrollment_id'],count4]
            index += 1
    else:
        df.iloc[index,:] = [ee['enrollment_id'].values[0],0]
        index += 1
    
df = df.sort('enrollment_id')

df.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/dffe24_3.csv',index=False)

df_ = pd.DataFrame(df.iloc[df['enrollment_id'].isin(dtrain['enrollment_id'].values).values,:]['in10days_max-to+1'].values,columns=['in10days_max-to+1'])
dtrain = pd.concat([dtrain,df_],axis=1)

df_ = pd.DataFrame(df.iloc[df['enrollment_id'].isin(dtest['enrollment_id'].values).values,:]['in10days_max-to+1'].values,columns=['in10days_max-to+1'])
dtest = pd.concat([dtest,df_],axis=1)

df = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/dffe24.csv')
df_ = pd.DataFrame(df.iloc[df['enrollment_id'].isin(dtrain['enrollment_id'].values).values,:][['in10days_enr','len_sum_in10days',u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'10','in20days_enr']].values,columns=['in10days_enr','len_sum_in10days','1_min_to+1','2_min_to+1','3_min_to+1','4_min_to+1','5_min_to+1','6_min_to+1','7_min_to+1','8_min_to+1','9_min_to+1','10_min_to+1','in20days_enr'])
dtrain = pd.concat([dtrain,df_],axis=1)

df_ = pd.DataFrame(df.iloc[df['enrollment_id'].isin(dtest['enrollment_id'].values).values,:][['in10days_enr','len_sum_in10days',u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'10','in20days_enr']].values,columns=['in10days_enr','len_sum_in10days','1_min_to+1','2_min_to+1','3_min_to+1','4_min_to+1','5_min_to+1','6_min_to+1','7_min_to+1','8_min_to+1','9_min_to+1','10_min_to+1','in20days_enr'])
dtest = pd.concat([dtest,df_],axis=1)





dtrain['in10days_enr+in20days_enr+in10days_max-to+1'] = dtrain['in10days_enr'] + dtrain['in20days_enr'] + dtrain['in10days_max-to+1']
dtest['in10days_enr+in20days_enr+in10days_max-to+1'] = dtest['in10days_enr'] + dtest['in20days_enr'] + dtest['in10days_max-to+1']

del dtrain['from'], dtrain['to']
del dtest['from'],dtest['to']


del dtrain['to+1'], dtrain['to+10']
del dtest['to+1'], dtest['to+10']



dtrain_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_2.csv')
dtest_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_2.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_2.csv',header=None)
label = label[0]

df = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/dffe24_3.csv')
df_ = pd.DataFrame(df.iloc[df['enrollment_id'].isin(dtrain['enrollment_id'].values).values,:]['in10days_max-to+1'].values,columns=['in10days_max-to+1'])
dtrain_leaf2 = pd.concat([dtrain_leaf2,df_],axis=1)

df_ = pd.DataFrame(df.iloc[df['enrollment_id'].isin(dtest['enrollment_id'].values).values,:]['in10days_max-to+1'].values,columns=['in10days_max-to+1'])
dtest_leaf2 = pd.concat([dtest_leaf2,df_],axis=1)

dtrain_leaf2['in10days_enr+in20days_enr+in10days_max-to+1'] = dtrain['in10days_enr'] + dtrain['in20days_enr'] + dtrain['in10days_max-to+1']
dtest_leaf2['in10days_enr+in20days_enr+in10days_max-to+1'] = dtest['in10days_enr'] + dtest['in20days_enr'] + dtest['in10days_max-to+1']


dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 3
for i in xrange(2,num_model,20):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(6,8),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    bst = pd.Series(bst).apply(f_)
    num_round_ = bst.argmax() + 380#60
    cc = bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe25%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe25%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)




#10日間の間の平日と休日の数
#date.csv追加(fe21.py追加）
course_date = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/date.csv')
f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
course_date['from'] = course_date['from'].apply(f_time)
course_date['to'] = course_date['to'].apply(f_time)
course_date = pd.concat([course_date,pd.DataFrame(np.zeros(len(course_date)*2).reshape(2,len(course_date)).T,columns=['WeekDay','HoliDay'])],axis=1)

for i in range(0,len(course_date)):
    W = 0
    H = 0
    for k in range(1,11):
        if date.fromordinal(course_date.iloc[i,:]['to']+k).weekday() in [0,1,2,3,4]:
            W += 1
        else:
            H += 1
    course_date.loc[i,['WeekDay']] = W
    course_date.loc[i,['HoliDay']] = H





course_date = course_date.merge(dtrain[['course_id','course_id_count']],on='course_id',how='inner')
course_date = course_date.drop_duplicates(['from','to','course_id_count'])
for i in range(0,len(course_date)):
    plt.hlines((i+1),course_date.iloc[i,:]['from'] ,course_date.iloc[i,:]['to'], linestyles="solid",linewidth=course_date.iloc[i,:]['course_id_count']/20104.0) 
plt.show()




'''
fe24_2.py
平日休日つける
'''


dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train4.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test4.csv')
dtrain = dtrain.sort('enrollment_id')
dtest = dtest.sort('enrollment_id')



#10日間の間の平日と休日の数
#date.csv追加(fe21.py追加）
course_date = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/date.csv')
f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
course_date['from'] = course_date['from'].apply(f_time)
course_date['to'] = course_date['to'].apply(f_time)
course_date = pd.concat([course_date,pd.DataFrame(np.zeros(len(course_date)*2).reshape(2,len(course_date)).T,columns=['WeekDay','HoliDay'])],axis=1)

for i in range(0,len(course_date)):
    W = 0
    H = 0
    for k in range(1,11):
        if date.fromordinal(course_date.iloc[i,:]['to']+k).weekday() in [0,1,2,3,4]:
            W += 1
        else:
            H += 1
    course_date.loc[i,['WeekDay']] = W
    course_date.loc[i,['HoliDay']] = H

dtrain = dtrain.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)
dtest = dtest.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)




dtrain_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_2.csv')
dtest_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_2.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_2.csv',header=None)
label = label[0]

dtrain_leaf2['WeekDay'] = dtrain['WeekDay']
dtrain_leaf2['HoliDay'] = dtrain['HoliDay']

dtest_leaf2['WeekDay'] = dtest['WeekDay']
dtest_leaf2['HoliDay'] = dtest['HoliDay']

dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 3
for i in xrange(2,num_model,20):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(6,8),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    bst = pd.Series(bst).apply(f_)
    num_round_ = bst.argmax() + 380#60
    cc = bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe25%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe25%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)


'''
fe24_2.py
間隔つける
'''
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')

label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
train = train.sort('enrollment_id')
test = test.sort('enrollment_id')
train.index = range(0,len(train))
test.index = range(0,len(test))
train = train.sort(['enrollment_id','time'])
test = test.sort(['enrollment_id','time'])


f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
train['time'] = train['time'].apply(f_time)
test['time'] = test['time'].apply(f_time)
#インターバル
def f_max(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.max()#, a.min(), a.mean(),

def f_min(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.min()#, a.mean(),

def f_mean(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.mean()

def f_std(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.std()

aa_train_max = train.groupby('enrollment_id')['time'].apply(f_max)
aa_test_max = test.groupby('enrollment_id')['time'].apply(f_max)
aa_train_min = train.groupby('enrollment_id')['time'].apply(f_min)
aa_test_min = test.groupby('enrollment_id')['time'].apply(f_min)
aa_train_mean = train.groupby('enrollment_id')['time'].apply(f_mean)
aa_test_mean = test.groupby('enrollment_id')['time'].apply(f_mean)
aa_train_std = train.groupby('enrollment_id')['time'].apply(f_std)
aa_test_std = test.groupby('enrollment_id')['time'].apply(f_std)

aa_train = pd.concat([aa_train_max,aa_train_min,aa_train_mean,aa_train_std],axis=1)
aa_test = pd.concat([aa_test_max,aa_test_min,aa_test_mean,aa_test_std],axis=1)




dtrain_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_2.csv')
dtest_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_2.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_2.csv',header=None)
label = label[0]

dtrain_leaf2 = pd.concat([dtrain_leaf2,pd.DataFrame(aa_train.values,columns=['interval_max','interval_min','interval_mean','interval_std'])],axis=1)
dtest_leaf2 = pd.concat([dtest_leaf2,pd.DataFrame(aa_test.values,columns=['interval_max','interval_min','interval_mean','interval_std'])],axis=1)


dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 3
for i in xrange(2,num_model,20):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(6,8),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    num_round_ = num_round + 380#60
    cc = 0.905000# bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe24_2_DT%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_2_DT%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)


'''
fe24_2.py
間隔つける
前半、中盤、後半
'''
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')

label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
train = train.sort('enrollment_id')
test = test.sort('enrollment_id')
train.index = range(0,len(train))
test.index = range(0,len(test))
train = train.sort(['enrollment_id','time'])
test = test.sort(['enrollment_id','time'])


f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
train['time'] = train['time'].apply(f_time)
test['time'] = test['time'].apply(f_time)
#インターバル
def f_max(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.max()#, a.min(), a.mean(),

def f_min(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.min()#, a.mean(),

def f_mean(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.mean()

def f_std(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.std()

aa_train_max = train.groupby('enrollment_id')['time'].apply(f_max)
aa_test_max = test.groupby('enrollment_id')['time'].apply(f_max)
aa_train_min = train.groupby('enrollment_id')['time'].apply(f_min)
aa_test_min = test.groupby('enrollment_id')['time'].apply(f_min)
aa_train_mean = train.groupby('enrollment_id')['time'].apply(f_mean)
aa_test_mean = test.groupby('enrollment_id')['time'].apply(f_mean)
aa_train_std = train.groupby('enrollment_id')['time'].apply(f_std)
aa_test_std = test.groupby('enrollment_id')['time'].apply(f_std)

aa_train = pd.concat([aa_train_max,aa_train_min,aa_train_mean,aa_train_std],axis=1)
aa_test = pd.concat([aa_test_max,aa_test_min,aa_test_mean,aa_test_std],axis=1)


#前半、中盤、後半
dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train4.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test4.csv')
dtrain = dtrain.sort('enrollment_id')
dtest = dtest.sort('enrollment_id')
label = dtrain['drop']

#date.csv追加(fe21.py追加）
course_date = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/date.csv')
f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
course_date['from'] = course_date['from'].apply(f_time)
course_date['to'] = course_date['to'].apply(f_time)
course_date['zenhan'] = course_date['from']+10
course_date['chuban'] = course_date['from']+20
course_date['kohan'] = course_date['to']

#dtest['zenhan'] = dtest['from']+10
#dtest['chuban'] = dtest['from']+20
#dtest['kohan'] = dtest['to']
train = train.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)
test = test.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)

train.index = range(0,len(train))
test.index = range(0,len(test))

#train,testのcourse_idでmerge
#dtrain = dtrain.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)
#dtest = dtest.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)

#dtrain.index = range(0,len(dtrain))
#dtest.index = range(0,len(dtest))


def f(data):
    zenhan = data.iloc[(data['time']<=data['zenhan']).values,:]
    #print data
    chuban = data.iloc[(data['zenhan']<data['time']).values,:]
    #print chuban
    chuban = chuban.iloc[(chuban['time']<=chuban['chuban']).values,:]
    kohan = data.iloc[(data['chuban']<data['time']).values,:]
    return pd.Series([len(zenhan),len(chuban),len(kohan)],index=['zenhan','chuban','kohan'])

bb_train_max = train.groupby('enrollment_id')[['time','zenhan','chuban','kohan']].apply(f)
bb_test_max = test.groupby('enrollment_id')[['time','zenhan','chuban','kohan']].apply(f)



dtrain_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_2.csv')
dtest_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_2.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_2.csv',header=None)
label = label[0]

dtrain_leaf2 = pd.concat([dtrain_leaf2,pd.DataFrame(aa_train.values,columns=['interval_max','interval_min','interval_mean','interval_std'])],axis=1)
dtest_leaf2 = pd.concat([dtest_leaf2,pd.DataFrame(aa_test.values,columns=['interval_max','interval_min','interval_mean','interval_std'])],axis=1)

dtrain_leaf2 = pd.concat([dtrain_leaf2,pd.DataFrame(bb_train_max.values,columns=['zenhan','chuban','kohan'])],axis=1)
dtest_leaf2 = pd.concat([dtest_leaf2,pd.DataFrame(bb_test_max.values,columns=['zenhan','chuban','kohan'])],axis=1)

#dtrain_leaf2['zenhan/sum_count'] = dtrain_leaf2['zenhan'] / dtrain_leaf2['sum_count']
#dtrain_leaf2['chuban/sum_count'] = dtrain_leaf2['chuban'] / dtrain_leaf2['sum_count']
#dtrain_leaf2['kohan/sum_count'] = dtrain_leaf2['kohan'] / dtrain_leaf2['sum_count']


dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 3
for i in xrange(2,num_model,20):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  3,
              "bst:max_depth" :  random.randint(6,8),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    num_round_ = num_round + 380#60
    cc = 0.905000# bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe24_2_DT%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_2_DT%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)




'''
fe24_3.py
間隔つける
前半、中盤、後半
event追加割合
'''
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')

label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
train = train.sort('enrollment_id')
test = test.sort('enrollment_id')
train.index = range(0,len(train))
test.index = range(0,len(test))
train = train.sort(['enrollment_id','time'])
test = test.sort(['enrollment_id','time'])


f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
train['time'] = train['time'].apply(f_time)
test['time'] = test['time'].apply(f_time)
#インターバル
def f_max(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.max()#, a.min(), a.mean(),

def f_min(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.min()#, a.mean(),

def f_mean(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.mean()

def f_std(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.std()

aa_train_max = train.groupby('enrollment_id')['time'].apply(f_max)
aa_test_max = test.groupby('enrollment_id')['time'].apply(f_max)
aa_train_min = train.groupby('enrollment_id')['time'].apply(f_min)
aa_test_min = test.groupby('enrollment_id')['time'].apply(f_min)
aa_train_mean = train.groupby('enrollment_id')['time'].apply(f_mean)
aa_test_mean = test.groupby('enrollment_id')['time'].apply(f_mean)
aa_train_std = train.groupby('enrollment_id')['time'].apply(f_std)
aa_test_std = test.groupby('enrollment_id')['time'].apply(f_std)

aa_train = pd.concat([aa_train_max,aa_train_min,aa_train_mean,aa_train_std],axis=1)
aa_test = pd.concat([aa_test_max,aa_test_min,aa_test_mean,aa_test_std],axis=1)


#前半、中盤、後半
dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train4.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test4.csv')
dtrain = dtrain.sort('enrollment_id')
dtest = dtest.sort('enrollment_id')
label = dtrain['drop']

#date.csv追加(fe21.py追加）
course_date = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/date.csv')
f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
course_date['from'] = course_date['from'].apply(f_time)
course_date['to'] = course_date['to'].apply(f_time)
course_date['zenhan'] = course_date['from']+10
course_date['chuban'] = course_date['from']+20
course_date['kohan'] = course_date['to']

#dtest['zenhan'] = dtest['from']+10
#dtest['chuban'] = dtest['from']+20
#dtest['kohan'] = dtest['to']
train = train.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)
test = test.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)

train.index = range(0,len(train))
test.index = range(0,len(test))

#train,testのcourse_idでmerge
#dtrain = dtrain.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)
#dtest = dtest.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)

#dtrain.index = range(0,len(dtrain))
#dtest.index = range(0,len(dtest))


def f(data):
    zenhan = data.iloc[(data['time']<=data['zenhan']).values,:]
    #print data
    chuban = data.iloc[(data['zenhan']<data['time']).values,:]
    #print chuban
    chuban = chuban.iloc[(chuban['time']<=chuban['chuban']).values,:]
    kohan = data.iloc[(data['chuban']<data['time']).values,:]
    return pd.Series([len(zenhan),len(chuban),len(kohan)],index=['zenhan','chuban','kohan'])

bb_train_max = train.groupby('enrollment_id')[['time','zenhan','chuban','kohan']].apply(f)
bb_test_max = test.groupby('enrollment_id')[['time','zenhan','chuban','kohan']].apply(f)

def f_event(data):
    #index = []
    zenhan = data.iloc[(data['time']<=data['zenhan']).values,:]
    zenhan_event = zenhan['event'].value_counts()#eventのカウント
    zenhan_event = zenhan_event/len(zenhan)#eventの割合
    zenhan_event.index += '_zenhan'
    #print data
    chuban = data.iloc[(data['zenhan']<data['time']).values,:]
    chuban = chuban.iloc[(chuban['time']<=chuban['chuban']).values,:]
    chuban_event = chuban['event'].value_counts()#eventのカウント
    chuban_event = chuban_event/len(chuban)#eventの割合
    chuban_event.index += '_chuban'

    kohan = data.iloc[(data['chuban']<data['time']).values,:]
    kohan_event = kohan['event'].value_counts()#eventのカウント
    kohan_event = kohan_event/len(kohan)#eventの割合
    kohan_event.index += '_kohan'
    #index = zenhan_event.index + chuban_event.index + kohan_event.index
    #print index
    zenhan_event = zenhan_event.append(chuban_event)
    zenhan_event = zenhan_event.append(kohan_event)
    #print zenhan_event
    return pd.Series(zenhan_event.values,index=zenhan_event.index)#,index=['zenhan','chuban','kohan'])

bb_train_event = train.groupby('enrollment_id')[['time','zenhan','chuban','kohan','event']].apply(f_event).unstack()
bb_test_event = test.groupby('enrollment_id')[['time','zenhan','chuban','kohan','event']].apply(f_event).unstack()




dtrain_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_2.csv')
dtest_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_2.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_2.csv',header=None)
label = label[0]

dtrain_leaf2 = pd.concat([dtrain_leaf2,pd.DataFrame(aa_train.values,columns=['interval_max','interval_min','interval_mean','interval_std'])],axis=1)
dtest_leaf2 = pd.concat([dtest_leaf2,pd.DataFrame(aa_test.values,columns=['interval_max','interval_min','interval_mean','interval_std'])],axis=1)

dtrain_leaf2 = pd.concat([dtrain_leaf2,pd.DataFrame(bb_train_max.values,columns=['zenhan','chuban','kohan'])],axis=1)
dtest_leaf2 = pd.concat([dtest_leaf2,pd.DataFrame(bb_test_max.values,columns=['zenhan','chuban','kohan'])],axis=1)

dtrain_leaf2 = pd.concat([dtrain_leaf2,pd.DataFrame(bb_train_event.values,columns=bb_train_event.columns)],axis=1)
dtest_leaf2 = pd.concat([dtest_leaf2,pd.DataFrame(bb_test_event.values,columns=bb_test_event.columns)],axis=1)


dtrain_leaf2.to_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_3.csv',index=False)
dtest_leaf2.to_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_3.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_3.csv',index=False)



dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 3
for i in xrange(2,num_model,20):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(6,8),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    num_round_ = num_round + 380#60
    cc = 0.905000# bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe24_2_DT%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_2_DT%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)


'''
fe24_4.py
間隔つける
前半、中盤、後半
event追加カウント
'''
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')

label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
#train = train.sort('enrollment_id')
test = test.sort('enrollment_id')
#train.index = range(0,len(train))
test.index = range(0,len(test))
#train = train.sort(['enrollment_id','time'])
test = test.sort(['enrollment_id','time'])


f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
#train['time'] = train['time'].apply(f_time)
test['time'] = test['time'].apply(f_time)
#インターバル
def f_max(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.max()#, a.min(), a.mean(),

def f_min(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.min()#, a.mean(),

def f_mean(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.mean()

def f_std(data):
    #print data
    #print data[1:len(data)].values
    #print data[:(len(data)-1)]
    if len(data) == 1:
        return 0
    a = data[1:len(data)].values - data[:(len(data)-1)].values
    #print a
    return a.std()

#aa_train_max = train.groupby('enrollment_id')['time'].apply(f_max)
aa_test_max = test.groupby('enrollment_id')['time'].apply(f_max)
#aa_train_min = train.groupby('enrollment_id')['time'].apply(f_min)
aa_test_min = test.groupby('enrollment_id')['time'].apply(f_min)
#aa_train_mean = train.groupby('enrollment_id')['time'].apply(f_mean)
aa_test_mean = test.groupby('enrollment_id')['time'].apply(f_mean)
#aa_train_std = train.groupby('enrollment_id')['time'].apply(f_std)
aa_test_std = test.groupby('enrollment_id')['time'].apply(f_std)

#aa_train = pd.concat([aa_train_max,aa_train_min,aa_train_mean,aa_train_std],axis=1)
aa_test = pd.concat([aa_test_max,aa_test_min,aa_test_mean,aa_test_std],axis=1)


#前半、中盤、後半
#dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train4.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test4.csv')
#dtrain = dtrain.sort('enrollment_id')
dtest = dtest.sort('enrollment_id')
#label = dtrain['drop']

#date.csv追加(fe21.py追加）
course_date = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/date.csv')
f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
course_date['from'] = course_date['from'].apply(f_time)
course_date['to'] = course_date['to'].apply(f_time)
course_date['zenhan'] = course_date['from']+10
course_date['chuban'] = course_date['from']+20
course_date['kohan'] = course_date['to']

#dtest['zenhan'] = dtest['from']+10
#dtest['chuban'] = dtest['from']+20
#dtest['kohan'] = dtest['to']
#train = train.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)
test = test.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)

#train.index = range(0,len(train))
test.index = range(0,len(test))

#train,testのcourse_idでmerge
#dtrain = dtrain.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)
#dtest = dtest.reset_index().merge(course_date,on='course_id',how='left').sort('index').drop('index',axis=1)

#dtrain.index = range(0,len(dtrain))
#dtest.index = range(0,len(dtest))


def f(data):
    zenhan = data.iloc[(data['time']<=data['zenhan']).values,:]
    #print data
    chuban = data.iloc[(data['zenhan']<data['time']).values,:]
    #print chuban
    chuban = chuban.iloc[(chuban['time']<=chuban['chuban']).values,:]
    kohan = data.iloc[(data['chuban']<data['time']).values,:]
    return pd.Series([len(zenhan),len(chuban),len(kohan)],index=['zenhan','chuban','kohan'])

#bb_train_max = train.groupby('enrollment_id')[['time','zenhan','chuban','kohan']].apply(f)
bb_test_max = test.groupby('enrollment_id')[['time','zenhan','chuban','kohan']].apply(f)

def f_event(data):
    #index = []
    zenhan = data.iloc[(data['time']<=data['zenhan']).values,:]
    zenhan_event = zenhan['event'].value_counts()#eventのカウント
    #zenhan_event = zenhan_event/len(zenhan)#eventの割合
    zenhan_event.index += '_zenhan'
    #print data
    chuban = data.iloc[(data['zenhan']<data['time']).values,:]
    chuban = chuban.iloc[(chuban['time']<=chuban['chuban']).values,:]
    chuban_event = chuban['event'].value_counts()#eventのカウント
    #chuban_event = chuban_event/len(chuban)#eventの割合
    chuban_event.index += '_chuban'

    kohan = data.iloc[(data['chuban']<data['time']).values,:]
    kohan_event = kohan['event'].value_counts()#eventのカウント
    #kohan_event = kohan_event/len(kohan)#eventの割合
    kohan_event.index += '_kohan'
    #index = zenhan_event.index + chuban_event.index + kohan_event.index
    #print index
    zenhan_event = zenhan_event.append(chuban_event)
    zenhan_event = zenhan_event.append(kohan_event)
    #print zenhan_event
    return pd.Series(zenhan_event.values,index=zenhan_event.index)#,index=['zenhan','chuban','kohan'])

#bb_train_event = train.groupby('enrollment_id')[['time','zenhan','chuban','kohan','event']].apply(f_event).unstack()
bb_test_event = test.groupby('enrollment_id')[['time','zenhan','chuban','kohan','event']].apply(f_event).unstack()

print bb_train_event


#dtrain_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_2.csv')
dtest_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_2.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_2.csv',header=None)
label = label[0]

#dtrain_leaf2 = pd.concat([dtrain_leaf2,pd.DataFrame(aa_train.values,columns=['interval_max','interval_min','interval_mean','interval_std'])],axis=1)
dtest_leaf2 = pd.concat([dtest_leaf2,pd.DataFrame(aa_test.values,columns=['interval_max','interval_min','interval_mean','interval_std'])],axis=1)

#dtrain_leaf2 = pd.concat([dtrain_leaf2,pd.DataFrame(bb_train_max.values,columns=['zenhan','chuban','kohan'])],axis=1)
dtest_leaf2 = pd.concat([dtest_leaf2,pd.DataFrame(bb_test_max.values,columns=['zenhan','chuban','kohan'])],axis=1)

#dtrain_leaf2 = pd.concat([dtrain_leaf2,pd.DataFrame(bb_train_event.values,columns=bb_train_event.columns)],axis=1)
dtest_leaf2 = pd.concat([dtest_leaf2,pd.DataFrame(bb_test_event.values,columns=bb_test_event.columns)],axis=1)


#dtrain_leaf2.to_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_4.csv',index=False)
dtest_leaf2.to_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_4.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_4.csv',index=False)



#dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 3
for i in xrange(2,num_model,20):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(6,8),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    num_round_ = num_round + 250#60
    cc = 0.905000# bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe24_2_DT%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_2_DT%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)



'''
fe24_3.py
'''
dtrain_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_3.csv')
dtest_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_3.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_3.csv',header=None)
label = label[0]


dtrain_leaf2 = dtrain_leaf2.fillna(0.000000001)
dtest_leaf2 = dtest_leaf2.fillna(0.000000001)


dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 30
for i in xrange(2,num_model,27):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(7,7),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    num_round_ = num_round + 250#60
    cc = 0.905000#bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe24_3fillna%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_3fillna%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)


'''
fe24_4.py
'''
dtrain_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_4.csv')
dtest_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_4.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_4.csv',header=None)
label = label[0]

dtrain_leaf2 = dtrain_leaf2.fillna(0.000000001)
dtest_leaf2 = dtest_leaf2.fillna(0.000000001)


dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 30
for i in xrange(2,num_model,27):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(7,7),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    num_round_ = num_round + 300#60
    cc = 0.905000#bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe24_4fillna%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_4fillna%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)


'''
fe24_3.py
'''
dtrain_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_3.csv')
dtest_leaf2 = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_3.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_3.csv',header=None)
label = label[0]


#dtrain_leaf2 = dtrain_leaf2.fillna(0.000000001)
#dtest_leaf2 = dtest_leaf2.fillna(0.000000001)


dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 30
for i in xrange(2,num_model,27):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(7,7),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    #bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    num_round_ = num_round + 250#60
    cc = 0.905000#bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe24_3%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_3%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)


'''
fe24_4.py(pred7)
'''
dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe24_4.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe24_4.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe24_4.csv',header=None)
label = label[0]

#dtrain_leaf2 = dtrain_leaf2.fillna(0.000000001)
#dtest_leaf2 = dtest_leaf2.fillna(0.000000001)
del dtest['2013_y_user_all_mean'],dtest['2014_y_user_all_mean'],dtest['10_m_y_user_all_mean'],dtest['11_m_y_user_all_mean'],dtest['12_m_y_user_all_mean'],dtest['01_m_y_user_all_mean'],dtest['02_m_y_user_all_mean'],dtest['05_m_y_user_all_mean'],dtest['06_m_y_user_all_mean'],dtest['07_m_y_user_all_mean'],dtest['08_m_y_user_all_mean'],dtrain['2013_10_user_all_new'],dtrain['2013_11_user_all_new'],dtrain['2013_12_user_all_new'],dtrain['2014_01_user_all_new'],dtrain['2014_02_user_all_new'],dtrain['2014_05_user_all_new'],dtrain['2014_06_user_all_new'],dtrain['2014_07_user_all_new'],dtrain['2014_08_user_all_new'],dtrain['2013_10_x_user_all_new_mean'],dtrain['2013_11_x_user_all_new_mean'],dtrain['2013_12_x_user_all_new_mean'],dtrain['2014_01_x_user_all_new_mean'],dtrain['2014_02_x_user_all_new_mean'],dtrain['2014_05_x_user_all_new_mean'],dtrain['2014_06_x_user_all_new_mean'],dtrain['2014_07_x_user_all_new_mean'],dtrain['2014_08_x_user_all_new_mean'],dtrain['2013_10_x_x_user_all_new_std'],dtrain['2013_11_x_x_user_all_new_std'],dtrain['2013_12_x_x_user_all_new_std'],dtrain['2014_01_x_x_user_all_new_std'],dtrain['2014_02_x_x_user_all_new_std'],dtrain['2014_05_x_x_user_all_new_std'],dtrain['2014_06_x_x_user_all_new_std'],dtrain['2014_07_x_x_user_all_new_std'],dtrain['2014_08_x_x_user_all_new_std']


del dtest['2013_y_user_all_std'],dtest['2014_y_user_all_std'],dtest['10_m_y_user_all_std'],dtest['11_m_y_user_all_std'],dtest['12_m_y_user_all_std'],dtest['01_m_y_user_all_std'],dtest['02_m_y_user_all_std'],dtest['05_m_y_user_all_std'],dtest['06_m_y_user_all_std'],dtest['07_m_y_user_all_std'],dtest['08_m_y_user_all_std'],dtest['2013_10_x_user_all_new_mean'],dtest['2013_11_x_user_all_new_mean'],dtest['2013_12_x_user_all_new_mean'],dtest['2014_01_x_user_all_new_mean'],dtest['2014_02_x_user_all_new_mean'],dtest['2014_05_x_user_all_new_mean'],dtest['2014_06_x_user_all_new_mean'],dtest['2014_07_x_user_all_new_mean'],dtest['2014_08_x_user_all_new_mean'],dtest['2013_10_x_x_user_all_new_std'],dtest['2013_11_x_x_user_all_new_std'],dtest['2013_12_x_x_user_all_new_std'],dtest['2014_01_x_x_user_all_new_std'],dtest['2014_02_x_x_user_all_new_std'],dtest['2014_05_x_x_user_all_new_std'],dtest['2014_06_x_x_user_all_new_std'],dtest['2014_07_x_x_user_all_new_std'],dtest['2014_08_x_x_user_all_new_std']

dtrain = dtrain.T.drop_duplicates().T
dtest = dtest.T.drop_duplicates().T


dtrain = xgb.DMatrix(dtrain, label=label)
dtest = xgb.DMatrix(dtest)


pred_name = []
num_model = 30
for i in xrange(2,num_model,27):
    random.seed((i+2)*254433433+234432443)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(7,7),
              "bst:eta" :  random.uniform(.01, .01),#step_size
              "bst:gamma" : round(random.uniform(0.5, 3.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,4.),
              "bst:subsample" :  round(random.uniform(.45, .6),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .52),2),
              "seed": 71
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(1300, 1300),0))
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

    bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
    #pred_prob = bst.predict( dtest )
    #print pred_prob
    #f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
    #bst = pd.Series(bst).apply(f_)
    num_round_ = num_round + 300#60
    cc = 0.905000#bst.max()
    if cc > 0.8980:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all_fe24_4%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_4%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)





#
pred1 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_2_DT2_0.905000_1680_7.csv',header=None)
pred2 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_3fillna29_0.905000_1550_7.csv',header=None)
pred3 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_4fillna2_0.905000_1600_7.csv',header=None)
pred4 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_4fillna29_0.905000_1600_7.csv',header=None)
pred5 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_32_0.905000_1550_7.csv',header=None)
pred6 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_329_0.905000_1550_7.csv',header=None)
pred7 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_42_0.905000_1600_7.csv',header=None)
pred8 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_allfe24_429_0.905000_1600_7.csv',header=None)



final_pred = (pred1[1]+pred2[1]+pred3[1]+pred4[1]+pred5[1]+pred6[1]+pred7[1]+pred8[1])/8.0
final_pred = pd.concat([pred1[0],final_pred],axis=1)
final_pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/PRED_ALL57.csv',header=None,index=False)



