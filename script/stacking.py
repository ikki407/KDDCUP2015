import graphlab as gl
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
from graphlab import feature_engineering as fe
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
import theano
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
import theano
import csv
from theano import tensor as T
import random
from nolearn.lasagne import BatchIterator
import lasagne.nonlinearities as nonlin
from lasagne.updates import adagrad, adadelta, sgd,rmsprop
from sklearn import metrics
#from lasagne.objectives import binary_crossentropy
from theano.tensor.nnet import binary_crossentropy
from lasagne.init import Constant, Glorot
from theano.tensor.nnet import sigmoid
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



dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train2.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test2.csv')
dtrain = dtrain.sort('enrollment_id')
dtest = dtest.sort('enrollment_id')
label = dtrain['drop']

'''
yyy = dtrain.groupby('username')['drop'].sum()
yyy = pd.DataFrame(yyy)
yyy['username'] = yyy.index


dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train.csv')
dtrain['sum_drop'] = dtrain.reset_index().merge(yyy,on='username',how='left').sort('index').drop('index',axis=1)['drop_y'].values
dtrain['sum_nondrop'] = dtrain['username_count'] - dtrain['sum_drop']
f_tonan = lambda x: 0 if x > 3 else np.nan
dtrain['sum_nondrop'] = dtrain['sum_nondrop'] - dtrain['username_count'].apply(f_tonan)
dtrain['sum_drop'] = dtrain['sum_drop'] - dtrain['username_count'].apply(f_tonan)

dtest['sum_drop'] = dtest.reset_index().merge(yyy,on='username',how='left').sort('index').drop('index',axis=1)['drop'].values
dtest['sum_nondrop'] = dtest['username_count'] - dtest['sum_drop']
dtest['sum_nondrop'] = dtest['sum_nondrop'] - dtest['username_count'].apply(f_tonan)
dtest['sum_drop'] = dtest['sum_drop'] - dtest['username_count'].apply(f_tonan)
'''
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

'''
#usernameの全体で見た情報を追加
xx = dtrain.groupby('username').sum()
for col in [u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg', u'sum_count']:
    dtrain[col+'_all_user'] = xx[col]

xx = dtest.groupby('username').sum()
for col in [u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg', u'sum_count']:
    dtest[col+'_all_user'] = xx[col]
'''









#クラスごとの離反率
aaa = dtrain.groupby('course_id')['drop'].sum()
aaa = pd.DataFrame(aaa,columns=['drop'])
aaa['course_id'] = aaa.index
aaa = aaa.sort('course_id')
bbb = dtrain['course_id'].value_counts()
bbb = pd.DataFrame(bbb,columns=['len'])
bbb['course_id'] = bbb.index
bbb = bbb.sort('course_id')
#aaa['drop'].values/bbb['len'].values.astype(float)
bbb['ratio_of_drop'] = aaa['drop'].values/bbb['len'].values.astype(float)

ccc = bbb[['course_id','ratio_of_drop']]
ccc['droprate_course'] = aaa['drop']#courseごとのdropout数
dtrain = dtrain.reset_index().merge(ccc,on='course_id',how='left').sort('index').drop('index',axis=1)
dtest = dtest.reset_index().merge(ccc,on='course_id',how='left').sort('index').drop('index',axis=1)
#dtrain = dtrain.sort('enrollment_id')
#dtest = dtest.sort('enrollment_id')
dtrain.index = range(0,len(dtrain))
dtest.index = range(0,len(dtest))

#username_countごとの離反率
aaa = dtrain.groupby('username_count')['drop'].sum()
aaa = pd.DataFrame(aaa,columns=['drop'])
aaa['username_count'] = aaa.index
aaa = aaa.sort('username_count')
bbb = dtrain['username_count'].value_counts()
bbb = pd.DataFrame(bbb,columns=['len'])
bbb['username_count'] = bbb.index
bbb = bbb.sort('username_count')
#aaa['drop'].values/bbb['len'].values.astype(float)
bbb['ratio_of_drop_uc'] = aaa['drop'].values/bbb['len'].values.astype(float)

ccc = bbb[['username_count','ratio_of_drop_uc']]
ccc['droprate_uc'] = aaa['drop']#username_countごとのdropout数
dtrain = dtrain.reset_index().merge(ccc,on='username_count',how='left').sort('index').drop('index',axis=1)
dtest = dtest.reset_index().merge(ccc,on='username_count',how='left').sort('index').drop('index',axis=1)
#dtrain = dtrain.sort('enrollment_id')
#dtest = dtest.sort('enrollment_id')
dtrain.index = range(0,len(dtrain))
dtest.index = range(0,len(dtest))


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

#del dtrain['enrollment_id'],dtrain['username'],dtrain['course_id'],
dtrain = pd.concat([b[:len(dtrain)],dtrain],axis=1)
#label = dtrain['drop']
#del dtrain['drop']
#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換


#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv')
#b = categorical(np.array(dtest['course_id']), drop=True)
#b = pd.DataFrame(b)
#dtest['username'] = aaa[0][len(dtrain):]

#dtest['course_id'] = bbb[0][len(dtrain):]



#del dtest['enrollment_id'],dtest['username'], dtest['course_id']

xx = b[len(dtrain):]
xx.index = range(0,len(xx))
dtest = pd.concat([xx,dtest],axis=1)
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換



#sum_countで割って, percentageを追加する
for col in [u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki']:
    dtrain[col+'_pct'] = dtrain[col]/dtrain[u'sum_count']

for col in [u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki']:
    dtest[col+'_pct'] = dtest[col]/dtest[u'sum_count']

dtrain['sum_count_cg'] = dtrain[[u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg']].sum(1)
#sum_countで割って, percentageを追加する
for col in [u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg']:
    dtrain[col+'_pct'] = dtrain[col]/dtrain[u'sum_count_cg']

dtest['sum_count_cg'] = dtest[[u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg']].sum(1)
#sum_countで割って, percentageを追加する
for col in [u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg']:
    dtest[col+'_pct'] = dtest[col]/dtest[u'sum_count_cg']


#year, month, day, hour ,weekdayのカウントデータを1にする
f_to1 = lambda x: 1 if x >=1 else 0
label_to1 = [u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w']




dtrain[label_to1] = dtrain[label_to1].applymap(f_to1)
dtest[label_to1] = dtest[label_to1].applymap(f_to1)

label_to1 = [u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki']
dtrain[label_to1] = dtrain[label_to1].applymap(f_to1)
dtest[label_to1] = dtest[label_to1].applymap(f_to1)

#label_to1 = [u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg']
#dtrain[label_to1] = dtrain[label_to1].applymap(f_to1)
#dtest[label_to1] = dtest[label_to1].applymap(f_to1)





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

#del dtrain['object_fact'], dtest['object_fact']

#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換
'''
numerics = dtrain.loc[:, [ u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg']]
 
# for each pair of variables, determine which mathmatical operators to use based on redundancy
for i in range(0, numerics.columns.size-1):
    for j in range(0, numerics.columns.size-1):
        col1 = str(numerics.columns.values[i])
        col2 = str(numerics.columns.values[j])
        # multiply fields together (we allow values to be squared)
        if i <= j:
            #name = col1 + "*" + col2
            #dtrain = pd.concat([dtrain, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
            name = col1 + "-" + col2
            dtrain = pd.concat([dtrain, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)
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
numerics = dtest.loc[:, [ u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg']]
# for each pair of variables, determine which mathmatical operators to use based on redundancy
for i in range(0, numerics.columns.size-1):
    for j in range(0, numerics.columns.size-1):
        col1 = str(numerics.columns.values[i])
        col2 = str(numerics.columns.values[j])
        # multiply fields together (we allow values to be squared)
        if i <= j:
            #name = col1 + "*" + col2
            #dtest = pd.concat([dtest, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
            name = col1 + "-" + col2
            dtest = pd.concat([dtest, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)
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
'''
onehot = fe.create(gl.SFrame(pd.DataFrame(train['object'])), fe.OneHotEncoder( max_categories=30))
def apply_onehot(dataset):
    if 'encoded_features' in dataset.column_names():
        # operation already completed on SFrame
        return dataset
    
    return onehot.transform(dataset)
qqq = apply_onehot(gl.SFrame(pd.DataFrame(train['object'])))
def f(x):
    try:
        return x.keys()[0]
    except:
        return 0
qqq2 = qqq.to_dataframe()
qqq2 = qqq2['encoded_features'].apply(f)
train['object'] = qqq2
qqq2 = train.groupby('enrollment_id')['object'].mean()
qqq2.index = range(0,len(qqq2))
dtrain_leaf['object_encoded'] = qqq2

#oneday = gl.SFrame(dtrain_leaf
hasher = fe.create(gl.SFrame(pd.DataFrame(dtrain['course_id'])), fe.FeatureHasher(num_bits=22))

def apply_feature_hasher(dataset):
    if 'hashed_features' in dataset.column_names():
        # feature hasher already performed
        return dataset
    return hasher.transform(dataset)
www = apply_feature_hasher(gl.SFrame(pd.DataFrame(dtrain['course_id'])))
f = lambda x: x.keys()[0]
www2 = www.to_dataframe()
www2 = www2['hashed_features'].apply(f)

dtrain['course_id'] = www2


del dtrain_leaf['start']
#2つのやまがあるのでそれぞれ分けて学習してみる
dtrain_leaf.iloc[(dtrain_leaf.loc[:,['max_time']]<350).values),:]
dtest_leaf.iloc[(dtest_leaf.loc[:,['max_time']]<350).values),:]

#dtrain_leaf['course_id'] = bbb[0][:len(dtrain)]
#dtest_leaf['course_id'] = bbb[0][len(dtrain):]
#ddd = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'sum_count', u'min_time', u'max_time', u'len_time',u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m']


#dtrain_leaf.loc[(dtrain_leaf['min_time']>350).values,'min_time'] = dtrain_leaf.loc[(dtrain_leaf['min_time']>350).values,'min_time'] - 198
#dtrain_leaf.loc[(dtrain_leaf['max_time']>350).values,'max_time'] = dtrain_leaf.loc[(dtrain_leaf['max_time']>350).values,'max_time'] - 198
#dtrain_leaf.loc[(dtrain_leaf['mean_time']>350).values,'mean_time'] = dtrain_leaf.loc[(dtrain_leaf['mean_time']>350).values,'mean_time'] - 198
dtrain['obj_not_nan'] = dtrain[[u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md']].sum(1)
dtest['obj_not_nan'] = dtest[[u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md']].sum(1)
dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/NN/train_NN.csv',index=False)
dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/label_NN.csv',index=False)
'''


using_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, u'username_count', u'course_id_count', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h',u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'0.0_md', u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md', u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg', u'sum_count', u'min_time', u'max_time', u'mean_time', u'len_time', u'max-mean', u'mean-min', u'access_pct', u'discussion_pct', u'nagivate_pct', u'page_close_pct', u'problem_pct', u'video_pct', u'wiki_pct',u'sum_count_cg', u'0_cg_pct', u'chapter_cg_pct', u'combinedopenended_cg_pct', u'problem_cg_pct', u'sequential_cg_pct', u'video_cg_pct', u'len_from_start']

dtrain['day_count'] = dtrain[[ u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d']].sum(1)

dtest['day_count'] = dtest[[ u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d']].sum(1)


Object = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/object.csv')
Object = Object.fillna('0')
Object = Object.drop_duplicates()
Object.index = range(0,len(Object))

#コースのstart時間
course_start = Object.loc[(Object['category']=='course').values,['course_id','start']]
f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
course_start['start'] = course_start['start'].apply(f_time)

dtrain = dtrain.reset_index().merge(course_start, on='course_id',how='left').sort('index').drop('index',axis=1)
dtrain['min-start'] = dtrain['min_time'] - dtrain['start']
dtrain['max-start'] = dtrain['max_time'] - dtrain['start']

dtest = dtest.reset_index().merge(course_start, on='course_id',how='left').sort('index').drop('index',axis=1)
dtest['min-start'] = dtest['min_time'] - dtest['start']
dtest['max-start'] = dtest['max_time'] - dtest['start']

#コースのchapter, sequentialなどの数
num_chapter = Object.groupby(['course_id','category']).count().unstack()['module_id']
num_chapter['course_id'] = num_chapter.index
dtrain['index'] = range(0,len(dtrain))
dtrain = dtrain.merge(num_chapter, on='course_id', how='left').sort('index').drop('index',axis=1)
dtrain = dtrain.fillna(0)
dtrain.index = range(0,len(dtrain))

dtest['index'] = range(0,len(dtest))
dtest = dtest.merge(num_chapter, on='course_id', how='left').sort('index').drop('index',axis=1)
dtest = dtest.fillna(0)
dtest.index = range(0,len(dtest))


#10日以内に新しいチャプターが始まるか
f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000 if x != 'null' else np.nan
chapter_time = Object.loc[(Object['category']=='chapter').values,['course_id','module_id','category','start']]
chapter_time['start'] = chapter_time['start'].apply(f_time)
chapter_time = chapter_time.dropna()

def in10days_num(x):
    course = x['course_id']
    max_time = x['max_time']
    course_chapter = chapter_time.iloc[(chapter_time['course_id']==course).values,:]['start']
    if True in pd.DataFrame([0 <= (course_chapter - max_time).values,(course_chapter - max_time).values <=10]).all().values:
        a = pd.DataFrame([0 <= (course_chapter - max_time).values,(course_chapter - max_time).values <=10]).all().value_counts()[True]#10日以内にあるかず
        b = 1#ダミー変数
        #print a
        #x['course_in10days'] = a
        #print x
        #x['course_in10days_dummy'] = b
        return a
    else:
        a = 0
        b = 0
        #x['course_in10days'] = a
        #x['course_in10days_dummy'] = b
        return a

def in10days_dummy(x):
    course = x['course_id']
    max_time = x['max_time']
    course_chapter = chapter_time.iloc[(chapter_time['course_id']==course).values,:]['start']
    if True in pd.DataFrame([0 <= (course_chapter - max_time).values,(course_chapter - max_time).values <=10]).all().values:
        a = pd.DataFrame([0 <= (course_chapter - max_time).values,(course_chapter - max_time).values <=10]).all().value_counts()[True]#10日以内にあるかず
        b = 1#ダミー変数
        #print a
        #x['course_in10days'] = a
        #print x
        #x['course_in10days_dummy'] = b
        return b
    else:
        a = 0
        b = 0
        #x['course_in10days'] = a
        #x['course_in10days_dummy'] = b
        return b

#10日以内にchapterが新しく始まるかの情報
#dtrain['course_in10days'] = dtrain.apply(in10days_num,axis=1)
#dtrain['course_in10days_dummy'] = dtrain.apply(in10days_dummy,axis=1)

#dtest['course_in10days'] = dtest.apply(in10days_num,axis=1)
#dtest['course_in10days_dummy'] = dtest.apply(in10days_dummy,axis=1)

'''
#その参加者が何回目の参加か
dtrain['drop'] = label
dtest.index = range(len(dtrain),len(dtrain)+len(dtest))
X = pd.concat([dtrain,dtest],ignore_index=True)
X = X.sort('min_time')
username = X['username'].drop_duplicates().values
X['past_enr'] = np.zeros(len(X))#過去の参加数
X['past_drop'] = np.zeros(len(X))#過去のdropout数
X['past_nondrop'] = np.zeros(len(X))#過去のdropout数
X['past_nondrop_rate'] = np.zeros(len(X))#過去のdropout率

new_X = pd.DataFrame()
count = 0
for i in username:
    username_data = X.loc[(X['username']==i).values,:]
    #%timeit 
    if len(username_data['drop'].values) > 1:#参加数が1回の人は欠損値とする
        a = username_data['drop']
        past_enr = range(1,len(a)+1)
        past_drop = a.cumsum()
        username_data['past_drop'] = past_drop.fillna(method='ffill')
    else:
        a = np.nan
        past_enr = 1
        past_drop = np.nan
        username_data['past_drop'] = np.nan
    username_data['past_enr'] = past_enr
    username_data['past_nondrop'] = username_data['past_enr'] - username_data['past_drop']
    username_data['past_nondrop_rate'] = username_data['past_nondrop']/username_data['past_enr']
    print username_data[['min_time','max_time','past_enr','past_drop','past_nondrop','past_nondrop_rate']]
    username_data.iloc[1:,:][['past_enr','past_drop','past_nondrop','past_nondrop_rate']] = username_data.iloc[:(len(username_data)-1),:][['past_enr','past_drop','past_nondrop','past_nondrop_rate']].values
    username_data.loc[username_data.index[0],['past_enr','past_drop','past_nondrop','past_nondrop_rate']] = [np.nan] *4
    
    new_X = pd.concat([new_X, username_data[['past_enr','past_drop','past_nondrop','past_nondrop_rate']]])
    count += 1
    if count % 1000 == 0:
        print len(username), count


past_dtrain = new_X.ix[dtrain.index]
#ss = new_X.ix[dtest.index]
#del ss['drop']
past_dtest = new_X.ix[dtest.index]
#保存
past_dtrain.to_csv('/Users/IkkiTanaka/Documents/KDDCup/train/new_train_past2.csv',index=False)
past_dtest.to_csv('/Users/IkkiTanaka/Documents/KDDCup/test/new_test_past2.csv',index=False)


dtrain = pd.concat([dtrain,past_dtrain],axis=1)
dtest = pd.concat([dtest,past_dtest],axis=1)

dtest.index = range(0,len(dtest))


dtrain = dtrain.sort_index()
dtest = dtest.sort_index()

label = dtrain['drop']
del dtrain['drop']

#X.to_csv('/Users/IkkiTanaka/Documents/KDDCup/enrollment_train_and_test.csv',index=False)
#X.groupby('username')


'''


dtrain[['past_enr','past_drop','past_nondrop','past_nondrop_rate']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/new_train_past2.csv')
dtest[['past_enr','past_drop','past_nondrop','past_nondrop_rate']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/new_test_past2.csv')

#dtrain.loc[(dtrain['past_enr']<=2).values,['past_drop','past_nondrop','past_nondrop_rate']] = np.nan
#dtest.loc[(dtest['past_enr']<=2).values,['past_drop','past_nondrop','past_nondrop_rate']] = np.nan

dtrain = dtrain.fillna(0)
dtest = dtest.fillna(0)

del dtrain['past_drop'], dtrain['past_nondrop']#, dtrain['past_nondrop_rate']
del dtest['past_drop'], dtest['past_nondrop']#, dtest['past_nondrop_rate']

'''
#そのusernameがその時点で何個のcourseに参加しているか

def count_course(time_array):
    a_min = []
    a_max = []
    for i in time_array:
        a_min.append(np.sum([1 for x in time_array if x[0] <= i[0] <= x[1]]))
        a_max.append(np.sum([1 for x in time_array if x[0] <= i[1] <= x[1]]))
    return np.vstack([a_min,a_max]).T

dtest.index = range(len(dtrain),len(dtrain)+len(dtest))
X = pd.concat([dtrain,dtest],ignore_index=True)
X = X.sort('min_time')
username = X['username'].drop_duplicates().values

new_X = pd.DataFrame()
count = 0
for i in username:
    username_data = X.loc[(X['username']==i).values,:]
    #%timeit 
    
    count_course_ = count_course(username_data[['min_time','max_time']].values)
    new_X = pd.concat([new_X, pd.DataFrame(count_course_, index=username_data.index, columns=['num_course_min','num_course_max'])])
    count += 1
    if count % 1000 == 0:
        print len(username), count


past_dtrain = new_X.ix[dtrain.index]
past_dtest = new_X.ix[dtest.index]
dtest.index = range(0,len(dtest))

dtrain = dtrain.sort_index()
dtest = dtest.sort_index()

#保存
past_dtrain.to_csv('/Users/IkkiTanaka/Documents/KDDCup/train/new_train_course_count.csv',index=False)
past_dtest.to_csv('/Users/IkkiTanaka/Documents/KDDCup/test/new_test_course_count.csv',index=False)
'''
#間違えたが精度上がった
#dtrain[['min_time','max_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/new_train_course_count.csv')
#dtest[['min_time','max_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/new_test_course_count.csv')

dtrain[['num_course_min','num_course_max']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/new_train_course_count.csv')
dtest[['num_course_min','num_course_max']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/new_test_course_count.csv')

#dtrain['num_course_min-max'] = dtrain['num_course_min'] - dtrain['num_course_max']
#dtrain['num_course_max-min'] = dtrain['num_course_max'] - dtrain['num_course_min']

#dtest['num_course_min-max'] = dtest['num_course_min'] - dtest['num_course_max']
#dtest['num_course_max-min'] = dtest['num_course_max'] - dtest['num_course_min']

#dtrain['num_course_min-max_abs'] = np.abs(dtrain['num_course_min'] - dtrain['num_course_max'])

#dtest['num_course_min-max_abs'] = np.abs(dtest['num_course_min'] - dtest['num_course_max'])


'''
#usernameをfactorize化
dtrain.loc[(dtrain['username_count']<10).values,'username'] = 'ikki'
dtest.loc[(dtest['username_count']<10).values,'username'] = 'ikki'

#a = dtrain.loc[(dtrain['username_count']>=10).values,:]
#a.iloc[(a['username']=='2oTvbzieHn2y5oozeOgSnruqE6N0BtR5').values,:]
#a2 = dtest.loc[(dtest['username_count']>=10).values,:]
#a2.iloc[(a2['username']=='2oTvbzieHn2y5oozeOgSnruqE6N0BtR5').values,:]

X = pd.concat([dtrain,dtest],ignore_index=True)
b = categorical(np.array(X['username']), drop=True)
b = pd.DataFrame(b)
dtrain = pd.concat([b[:len(dtrain)],dtrain],axis=1)
xx = b[len(dtrain):]
xx.index = range(0,len(xx))
dtest = pd.concat([xx,dtest],axis=1)
'''





del dtrain['username'],dtrain['course_id'],dtrain['enrollment_id'],
del dtest['username'], dtest['course_id'],dtest['enrollment_id'],

#学期の始まりからの時間
f_from_start = lambda x: x - 168 if x < 350 else x - 366
dtrain['len_from_start'] = dtrain['min_time'].apply(f_from_start)
dtest['len_from_start'] = dtest['min_time'].apply(f_from_start)

#学期の終わりからの時間
f_from_end = lambda x: 279 - x if x < 350 else 446 - x
dtrain['len_from_end'] = dtrain['max_time'].apply(f_from_end)
dtest['len_from_end'] = dtest['max_time'].apply(f_from_end)

#一日毎の平均log回数
dtrain['mean_stay_times'] = dtrain['sum_count']/dtrain['day_count']
dtest['mean_stay_times'] = dtest['sum_count']/dtest['day_count']

#一日毎の平均event回数
for col in [u'browser', u'server', u'access', u'discussion_x', u'nagivate', u'page_close', u'problem_x', u'video_x', u'wiki']:
    dtrain[col+'_per_day'] = dtrain[col]/dtrain[u'day_count']

for col in [u'browser', u'server', u'access', u'discussion_x', u'nagivate', u'page_close', u'problem_x', u'video_x', u'wiki']:
    dtest[col+'_per_day'] = dtest[col]/dtest[u'day_count']


#dtrain['len_from_end2'] = dtrain['min_time'].apply(f_from_end)
#dtest['len_from_end2'] = dtest['min_time'].apply(f_from_end)

#label = dtrain['drop']
#del dtrain['drop']

dtrain['2013_10'] = dtrain['10_m']
dtrain['2013_11'] = dtrain['11_m']
dtrain['2013_12'] = dtrain['12_m']
dtrain['2014_01'] = dtrain['01_m']
dtrain['2014_02'] = dtrain['02_m']
dtrain['2014_05'] = dtrain['05_m']
dtrain['2014_06'] = dtrain['06_m']
dtrain['2014_07'] = dtrain['07_m']
dtrain['2014_08'] = dtrain['08_m']

del dtrain['2013'],dtrain['2014'],dtrain['10_m'],dtrain['11_m'],dtrain['12_m'],dtrain['01_m'],dtrain['02_m'],dtrain['05_m'],dtrain['06_m'],dtrain['07_m'],dtrain['08_m']

dtest['2013_10'] = dtest['10_m']
dtest['2013_11'] = dtest['11_m']
dtest['2013_12'] = dtest['12_m']
dtest['2014_01'] = dtest['01_m']
dtest['2014_02'] = dtest['02_m']
dtest['2014_05'] = dtest['05_m']
dtest['2014_06'] = dtest['06_m']
dtest['2014_07'] = dtest['07_m']
dtest['2014_08'] = dtest['08_m']

del dtest['2013'],dtest['2014'],dtest['10_m'],dtest['11_m'],dtest['12_m'],dtest['01_m'],dtest['02_m'],dtest['05_m'],dtest['06_m'],dtest['07_m'],dtest['08_m']


dtrain.index = range(0,len(dtrain))
dtest.index = range(0,len(dtest))

'''
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=50, init='k-means++', n_init=40, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=1, random_state=1992, copy_x=True, n_jobs=-1)
clf.fit(dtrain)
dtrain['kmeans50'] = clf.predict(dtrain)
dtest['kmeans50'] = clf.predict(dtest)
clf = KMeans(n_clusters=40, init='k-means++', n_init=40, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=1, random_state=1992, copy_x=True, n_jobs=-1)
clf.fit(dtrain)
dtrain['kmeans40'] = clf.predict(dtrain)
dtest['kmeans40'] = clf.predict(dtest)
clf = KMeans(n_clusters=30, init='k-means++', n_init=40, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=1, random_state=1992, copy_x=True, n_jobs=-1)
clf.fit(dtrain)
dtrain['kmeans30'] = clf.predict(dtrain)
dtest['kmeans30'] = clf.predict(dtest)
clf = KMeans(n_clusters=20, init='k-means++', n_init=40, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=1, random_state=1992, copy_x=True, n_jobs=-1)
clf.fit(dtrain)
dtrain['kmeans20'] = clf.predict(dtrain)
dtest['kmeans20'] = clf.predict(dtest)
clf = KMeans(n_clusters=10, init='k-means++', n_init=40, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=1, random_state=1992, copy_x=True, n_jobs=-1)
clf.fit(dtrain)
dtrain['kmeans10'] = clf.predict(dtrain)
dtest['kmeans10'] = clf.predict(dtest)
'''







dtrain_leaf = dtrain
dtest_leaf = dtest
print dtrain_leaf.columns == dtest_leaf.columns
aa = dtrain_leaf.columns

#dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_train.csv',index=False)
#dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_test.csv',index=False)
#label.to_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_label.csv',index=False)

#f_tonan = lambda x: np.nan if x == 0 else x
#dtrain_leaf = dtrain_leaf.applymap(f_tonan)


n_val = round(len(dtrain)*0.5)
np.random.seed(19920407)
rows = np.random.choice(dtrain.index.values, len(dtrain),replace=False)
dtrain_leaf1 = dtrain_leaf.ix[rows[n_val:]]
label1 = label[rows[n_val:]]

dtrain_leaf2 = dtrain_leaf.ix[rows[:n_val]]
label2 = label[rows[:n_val]]

dtrain1 = xgb.DMatrix(dtrain_leaf1, label=label1)
dtrain2 = xgb.DMatrix(dtrain_leaf2, label=label2)

dtest = xgb.DMatrix(dtest_leaf)


random.seed(25433+23)
param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(4,4),
              "bst:eta" :  random.uniform(.01, .05),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0.,4.),
              "bst:subsample" :  round(random.uniform(.3, .9),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.4, 1.),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .55),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
num_round = int( round(random.uniform(300, 350),0))
print param
print num_round
evallist  = [(dtrain2, 'eval'),(dtrain1,'train')]
bst = xgb.train(param,dtrain1, num_round,evallist)
pred_prob2 = bst.predict( dtrain2 )

dtrain_leaf2['pred'] = pred_prob2

evallist  = [(dtrain1, 'eval'),(dtrain2,'train')]
bst = xgb.train(param,dtrain2, num_round,evallist)
pred_prob1 = bst.predict( dtrain1 )

dtrain_leaf1['pred'] = pred_prob1

dtrain_leaf_all = pd.concat([dtrain_leaf1,dtrain_leaf2])
label = pd.concat([label1,label2])
dtrain_all = xgb.DMatrix(dtrain_leaf_all, label=label)

random.seed(25433+23)
param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(6,6),
              "bst:eta" :  random.uniform(.01, .05),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0.,4.),
              "bst:subsample" :  round(random.uniform(.3, .9),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.4, 1.),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .55),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
print param
print num_round
num_round = int( round(random.uniform(500, 550),0))

evallist  = [(dtrain_all,'train')]
bst=xgb.cv(param, dtrain_all, num_round, nfold=5, seed = 1992, show_stdv = False)
bst = xgb.train(param,dtrain_all, num_round,evallist)
pred_prob_test = bst.predict( dtest )

dtest_leaf['pred'] = pred_prob_test

dtrain = xgb.DMatrix(dtrain_leaf_all, label=label)
dtest = xgb.DMatrix(dtest_leaf)


pred_name = []
num_model = 30
for i in xrange(0,num_model):
    random.seed(i*25433+23)
    param = {"objective" : "binary:logistic",#"binary:logistic",
              "eval_metric" :  'auc',
              #"eval_metric" : "ams@0.15",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(2,2),
              "bst:eta" :  random.uniform(.01, .05),#step_size
              "bst:gamma" : round(random.uniform(.0, 8.),1),#min_loss_reduction
              "bst:min_child_weight" : random.randint(0.,4.),
              "bst:subsample" :  round(random.uniform(.3, .9),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.4, 1.),2),#column_subsample
              "silent": 1,
              "base_score": round(random.uniform(.50, .55),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 1.6,
              }
    #if random.randint(0,1):
    #    param['scale_pos_weight'] = scale_pos_weight
    num_round = int( round(random.uniform(500, 550),0))
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
    num_round_ = bst.argmax() + 100#60
    cc = bst.max()
    if cc > 0.8907:
        watchlist  = [(dtrain,'train')]
        np.random.seed(19920407)
        bst = xgb.train(param,dtrain, num_round_,watchlist)
        pred_prob = bst.predict( dtest )
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        pred_name.append('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all4%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]))
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all4%i_%f_%i_%i.csv' % (i,cc,num_round_, param["bst:max_depth"]) ,header=None,index=False)

Pred = pd.DataFrame(np.zeros(80362),columns=[1])[1]
for i in xrange(0,len(pred_name)):
    pred = pd.read_csv(pred_name[i], header=None)
    Pred = Pred + pred[1]
Pred = Pred/float(len(pred_name))
Pred = pd.concat([pred[0],Pred],axis=1)
Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred46.csv',header=None,index=False)



