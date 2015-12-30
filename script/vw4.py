# -*- coding: UTF-8 -*-

###################################
#feature engineering
#vowpal_wabbit
###################################
import hashlib
import argparse, csv, sys, pickle, collections, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
from collections import Counter
import pandas as pd
from datetime import datetime
from csv import DictReader
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
import matplotlib.pyplot as plt
import math
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
cd /Users/IkkiTanaka/Documents/KDDCup

#log transform
f = lambda x: np.log(1+x**2) if (x > 1 and (type(x) == np.int64 or type(x) == np.float64)) else x



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

del dtrain['enrollment_id']#dtrain['course_id'],dtrain['username'],

#label = dtrain['drop']
#del dtrain['drop']
#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換


#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv')
#b = categorical(np.array(dtest['course_id']), drop=True)
#b = pd.DataFrame(b)
#dtest['username'] = aaa[0][len(dtrain):]

#dtest['course_id'] = bbb[0][len(dtrain):]



del dtest['enrollment_id']#dtest['course_id'],dtest['username'],

#course_idのファクトライズ化
#dtrain = pd.concat([b[:len(dtrain)],dtrain],axis=1)
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

#train_time.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv',index=False)
#test_time.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv',index=False)

#dtrain['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
#dtest['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/test_timelen.csv')

'''

#train_course = dtrain['course_id'] +1
#test_course = dtest['course_id'] +1


#dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')

#dtrain = dtrain.iloc[:,39:]
#dtest = dtest.iloc[:,39:]

#f_to0 = lambda x: -1 if x == 0 else x
#dtrain = dtrain.applymap(f_to0)
#dtest = dtest.applymap(f_to0)

#dtrain = dtrain.applymap(f)#対数変換
#dtest = dtest.applymap(f)#対数変換

'''
[u'username', u'course_id',u'username_count', u'course_id_count', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'sum_count']
'''
using_label =[u'username', u'course_id',u'username_count', u'course_id_count', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'sum_count']

dtrain2 = dtrain[using_label]
dtest2 = dtest[using_label]


dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/train_NN.csv',index=False)
dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/label_NN.csv',index=False)


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
!vw fe_train/fe_vw/new_enr_train2.vw -k -c -f fe_train/fe_vw/train.model2.vw --loss_function logistic --passes 30 -l 0.2 -b 30 --holdout_off --random_seed 19920407 --decay_learning_rate 0.8 --rank 10

#--nn 30 -q ii
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


