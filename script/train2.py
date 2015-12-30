import sys
import pandas as pd
import numpy as np
import os
import hashlib
import argparse, csv, sys, pickle, collections, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
import matplotlib.pyplot as plt


#set path
sys.path.append('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction')
sys.path.append('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/script')

#set directory
os.chdir('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction')

#read data
coupon_area_train = pd.read_csv('data/coupon_area_train.csv')#the coupon listing area for the training set coupons（クーポンのエリア）
coupon_area_test = pd.read_csv('data/coupon_area_test.csv')#the coupon listing area for the testing set coupons（クーポンのエリア）

coupon_list_train = pd.read_csv('data/coupon_list_train.csv')#the master list of coupons（訓練クーポンのマスター）
coupon_list_test = pd.read_csv('data/coupon_list_test.csv')#the master list of coupons.predictions should be sourced only from these 310 coupons（テストクーポンのマスター）

coupon_visit_train = pd.read_csv('data/coupon_visit_train.csv')#the viewing log of users browsing coupons during the training set time period. not provided this table for the test set period（クーポンの閲覧ログ）
coupon_detail_train = pd.read_csv('data/coupon_detail_train.csv')#the purchase log of users buying coupons during the training set time period. not provided this table for the test set period（クーポンの購入ログ）

user_list = pd.read_csv('data/user_list.csv')#the master list of users in the dataset（ユーザーのリスト、予測すべきユーザー）
sample_submission = pd.read_csv('sample_submission.csv')

#trainデータを分析可能となるように加工
#trainデータ
label = [u'REG_DATE', u'SEX_ID', u'AGE', u'WITHDRAW_DATE', u'PREF_NAME', u'USER_ID_hash',u'VIEW_COUPON_ID_hash',u'PURCHASE_FLG']
train = user_list.merge(coupon_visit_train,on=u'USER_ID_hash',how='outer')[label]
train = train.merge(coupon_list_train,left_on=u'VIEW_COUPON_ID_hash',right_on=u'COUPON_ID_hash',how='left')
train = train.ix[train['COUPON_ID_hash'].dropna().index]
train.index = range(0,len(train))
del train[u'VIEW_COUPON_ID_hash']


#LibFM用のデータに変換

#hash関数
NR_BINS = 1000000000
def hashstr(input):
    #if type(input)
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

#convert関数
fields = [u'REG_DATE', u'SEX_ID', u'AGE', u'WITHDRAW_DATE', u'PREF_NAME', u'USER_ID_hash', u'PURCHASE_FLG', u'CAPSULE_TEXT', u'GENRE_NAME', u'PRICE_RATE', u'CATALOG_PRICE', u'DISCOUNT_PRICE', u'DISPFROM', u'DISPEND', u'DISPPERIOD', u'VALIDFROM', u'VALIDEND', u'VALIDPERIOD', u'USABLE_DATE_MON', u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT', u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY', u'large_area_name', u'ken_name', u'small_area_name', u'COUPON_ID_hash']


#TrainのLibFMデータ作成（シャッフルせよ）
label = [u'REG_DATE', u'SEX_ID', u'AGE', u'WITHDRAW_DATE', u'PREF_NAME', u'USER_ID_hash', u'CAPSULE_TEXT', u'GENRE_NAME', u'PRICE_RATE', u'CATALOG_PRICE', u'DISCOUNT_PRICE', u'DISPFROM', u'DISPEND', u'DISPPERIOD', u'VALIDFROM', u'VALIDEND', u'VALIDPERIOD', u'USABLE_DATE_MON', u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT', u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY', u'large_area_name', u'ken_name', u'small_area_name', u'COUPON_ID_hash']
train = train.fillna('NaN')

#trainの中のdtype==objectをhash関数に入れる
label_object = train.dtypes[train.dtypes == object].index
new_train = pd.DataFrame()
new_train[label_object] = train[label_object].applymap(hashstr) + [':1'] * len(label_object)

label_float = train.dtypes[train.dtypes != object].index
new_train[label_float] = train[label_float].applymap(hashstr) + [':1'] * len(label_object)

new_train = new_train.iloc[np.random.permutation(len(new_train))]
new_train.index = range(0,len(new_train))


def train_convert(train1,parchase, dst_path, is_train):
    with open(dst_path, 'w') as f:
        #for row in csv.DictReader(open(src_path)):
        num = 0
        for row in train1.values:
            #row = train1.iloc[index,:]

            if is_train == True:
                f.write('{0} {1}\n'.format(parchase[num], ' '.join(row)))
            if is_train == False:
                f.write('{0} {1}\n'.format(0, ' '.join(feats)))
            num += 1
            if num % 10000 == 0:
                print num



train_convert(new_train,train[u'PURCHASE_FLG'].values,'/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1.fm',True)


#TestデータのLibFMのデータ作成
new_user_list = ['0:','1:','2:','3:','4:','5:'] + user_list.astype(str).applymap(hashstr)
new_coupon_list_test = ['6:','7:','8:','9:','10:','11:','12:','13:','14:','15:','16:','17:','18:','19:','20:','21:','22:','23:','24:','25:','26:','27:','28:','29:'] + coupon_list_test.astype(str).applymap(hashstr)

def test_convert(user_list1,coupon_list_test1,dst_path,is_train):
    with open(dst_path, 'w') as f:
        #for row in csv.DictReader(open(src_path)):
        num = 0
        for row1 in user_list1.values:
            for row2 in coupon_list_test1.values:
                row = list(row1)
                row.extend(list(row2))
            #row = train1.iloc[index,:]

                if is_train == True:
                    f.write('{0} {1}\n'.format(parchase[num], ' '.join(row)))
                if is_train == False:
                    f.write('{0} {1}\n'.format(0, ' '.join(row)))
                num += 1
                #print num
                if num % 100000 == 0:
                    print num

#user_ID_hash保存
def test_convert_user_id_hash(user_list1,coupon_list_test1,dst_path):
    with open(dst_path, 'w') as f:
        #for row in csv.DictReader(open(src_path)):
        num = 0
        for row1 in user_list1.values:
            for row2 in coupon_list_test1.values:
                f.write('{0},\n'.format(row1))
                num += 1
                #print num
                if num % 100000 == 0:
                    print num

test_convert(new_user_list,new_coupon_list_test,'/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_test.fm',False)

test_convert_user_id_hash(user_list[u'USER_ID_hash'],new_coupon_list_test,'/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_test_user_id_hsash.csv')

!/Users/IkkiTanaka/libfm-master/bin/libFM -dim '1,1,16' -iter 15 -learn_rate 0.05 -method mcmc -init_stdev 2.0 -task c -verbosity 2 -regular '0.01,0,0.01' -train /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1.fm -out /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction.csv -test /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_test.fm


c=pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction.csv',header=None,names=['probability'])

c['USER_ID_hash'] = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_test_user_id_hsash.csv',header=None)[0].values
c['COUPON_ID_hash'] =  list(coupon_list_test['COUPON_ID_hash'].values) * len(user_list)

def top10(data):
    data = data.sort('probability',ascending=False)
    #print data.columns
    top10 = data['COUPON_ID_hash'].values[:10]
    #print top10
    user = data['USER_ID_hash'].values[0]
    #print user
    return pd.Series([user,' '.join(top10)],index=[u'USER_ID_hash',u'PURCHASED_COUPONS'])

pred = c.groupby('USER_ID_hash').apply(top10)
pred = pd.DataFrame(pred.values,columns=[u'USER_ID_hash',u'PURCHASED_COUPONS'])
pred.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/FM1.csv',index=False)

index = 0
index2 = 0
for i in user_list['USER_ID_hash'].values:
    c.iloc[(0+index),1] = i
    index += 310
    index2 += 1
    if index2 % 100 == 0:
        print index2

user_list['USER_ID_hash'].values

coupon_list_test['COUPON_ID_hash']

