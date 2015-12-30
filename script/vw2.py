# -*- coding: UTF-8 -*-

###################################
#feature engineering
#vowpal_wabbit
###################################

'''
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



'''

import pandas as pd
from datetime import datetime
from csv import DictReader
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
import matplotlib.pyplot as plt
import math

label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)

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
        if k not in ['enrollment_id']:
          if k in [u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'sum_count']: # numerical feature, example: I5
            if len(str(v)) > 0: #check for empty values
              numerical_features += " %s:%s" % (k,v)
          if k in [u'username', u'course_id']: # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v
			  
	  #Creating the labels		  
      if train: #we care about labels
        if int(labels[labels[0]==int(row['enrollment_id'])][1]) == 1:
          label = 1
        else:
          label = -1 #we set negative label to -1
        outfile.write( "%s |i%s |c%s\n" % (label, numerical_features, categorical_features) )
		
      else: #we dont care about labels
        outfile.write( "1 |i%s |c%s\n" % (numerical_features, categorical_features) )
      
	  #Reporting progress
      if e % 10000 == 0:
        print("%s\t%s"%(e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))




#train, validation
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/fe_train/new_enr_train.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/new_enr_train.vw',train=True,labels=label)
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/fe_train/enr_validation.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/enr_validation.vw',train=True,labels=label)

#shuffle train
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_vw/shuffle_enr_train.vw',train=True,labels=label)

#test
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_vw/new_enr_test.vw',train=False)



#学習
cd /Users/IkkiTanaka/Documents/KDDCup
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function logistic --passes 1 -l 3.0 -b 30 --holdout_off --cubic ccc 

!vw fe_train/fe_vw/enr_validation.vw -t -i fe_train/fe_vw/train.model1.vw -p fe_test/fe_vw/test1_val.txt

#!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function quantile --passes 1 -l 3.0 -b 30 --holdout_off --cubic ccc --quantile_tau 0.38 --noconstant --random_seed 19920407


#!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function hinge --passes 1 -l 5.0 -b 30 --holdout_off --cubic ccc


outputfile = "/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/pred1_val.csv" 

def zygmoid(x):
    return 1 / (1 + math.exp(-x))

with open(outputfile,"wb") as outfile:
    #outfile.write("Id,Predicted\n")
    for line in open("fe_test/fe_vw/test1_val.txt"):
        row = line.strip().split(" ")
        outfile.write("%f\n"%(zygmoid(float(row[0]))))

#ROC
pred_val = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/pred1_val.csv',header=None)

y = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/val_enr_label.csv',header=None)
print pd.concat([y[0],pred_val[0]],axis=1)
fpr, tpr, thresholds = roc_curve(y, pd.DataFrame(list(pred_val[0])), pos_label=1)  
print auc(fpr, tpr)
plt.plot(fpr,tpr)
plt.show()


#テスト
!vw fe_test/fe_vw/new_enr_test.vw -t -i fe_train/fe_vw/train.model1.vw -p fe_test/fe_vw/test1.txt

outputfile = "/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/pred.csv" 
with open(outputfile,"wb") as outfile:
    #outfile.write("Id,Predicted\n")
    for line in open("fe_test/fe_vw/test1.txt"):
        row = line.strip().split(" ")
        outfile.write("%f\n"%(zygmoid(float(row[0]))))

pred = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/pred.csv',header=None)
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)

Pred = pd.concat([sample[0],pred[0]],axis=1)

Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_vw/pred13.csv',header=None,index=False)


'''
pred1
0.863907226522(val)
0.8368917891944319(LB)
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function logistic --passes 1 -l 0.1 -b 30 --holdout_off

pred2
0.889408277247
0.8442281328216192
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function logistic --passes 1 -l 0.3 -b 30 --holdout_off

pred3
0.903837083381
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function logistic --passes 1 -l 0.5 -b 30 --holdout_off

pred4
0.91252313147
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function logistic --passes 1 -l 0.7 -b 30 --holdout_off

pred5
0.919932189023
0.8391296897919682
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function logistic --passes 1 -l 1.0 -b 30 --holdout_off

pred6
0.928629303954
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function logistic --passes 1 -l 3.0 -b 30 --holdout_off

pred7
0.955991626507
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function logistic --passes 1 -l 3.0 -b 30 --holdout_off -q cc

pred8
0.965546176761
0.8322797541061842
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function logistic --passes 1 -l 3.0 -b 30 --holdout_off --cubic ccc

pred9
0.965176719326
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function quantile --passes 1 -l 3.0 -b 30 --holdout_off --cubic ccc

pred10
0.966199386134
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function quantile --passes 1 -l 3.0 -b 30 --holdout_off --cubic ccc --quantile_tau 0.2

pred11
0.966803795114
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function quantile --passes 1 -l 3.0 -b 30 --holdout_off --cubic ccc --quantile_tau 0.35

pred12
0.966849432809
!vw fe_train/fe_vw/new_enr_train.vw -k -c -f fe_train/fe_vw/train.model1.vw --loss_function quantile --passes 1 -l 3.0 -b 30 --holdout_off --cubic ccc --quantile_tau 0.38  --noconstant



'''
