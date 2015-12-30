#################################################################
# log_trainを用いたFFM
#################################################################

import hashlib
import argparse, csv, sys, pickle, collections, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np


'''
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup//test/log_test.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']

train = train.merge(label, how='inner',on='enrollment_id')
f = lambda x: x[:10]
train['time'] = train['time'].apply(f)
test['time'] = test['time'].apply(f)

#時間はhourも見たほうが良いかも！？


X = pd.concat((train[[col for col in train.columns if col in ["enrollment_id",'username','course_id','time','source','event','object'] ]], test))
count_lk = {col: X.groupby(col).aggregate({"enrollment_id": "count"}).to_dict()["enrollment_id"] for col in X.columns}

for col in ["enrollment_id",'course_id','source','event','object']:
    train[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    test[col + "_count"] = ([count_lk[col][x] for x in test[col]])

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

'''
'''
1. trainをsplitする8:2

train: new_train.csv
validation: validation.csv -> label: val_label

2. all_dataはtimeを変換して
shuffle_train.csv
new_test

'''





NR_BINS = 1000000
def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)


fields = ['username','course_id','time','source','event','object', u'enrollment_id_count', u'course_id_count', u'source_count', u'event_count', u'object_count', u'username_count']

label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)

def convert(src_path, dst_path, is_train):
    with open(dst_path, 'w') as f:
        for row in csv.DictReader(open(src_path)):
            i = 1
            w = 1
            feats = []

            for field in fields:
                v = hashstr(field+'-'+row[field])
                #print v
                feats.append('{i}:{v}:{w}'.format(i=i, v=v, w=w))
                i += 1
            #print row

            if is_train == True:
                f.write('{0} {1}\n'.format(int(label[label[0]==int(row['enrollment_id'])][1]), ' '.join(feats)))
            if is_train == False:
                f.write('{0} {1}\n'.format(0, ' '.join(feats)))


convert('/Users/IkkiTanaka/Documents/KDDCup/train/new_train.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/ffm/new_train.ffm', True)
convert('/Users/IkkiTanaka/Documents/KDDCup/train/validation.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/ffm/validation.ffm', True)
convert('/Users/IkkiTanaka/Documents/KDDCup/test/new_test.csv', '/Users/IkkiTanaka/Documents/KDDCup/test/ffm/new_test.ffm', False)
convert('/Users/IkkiTanaka/Documents/KDDCup/train/shuffle_train.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/ffm/shuffle_train.ffm', True)


#学習
!/Users/IkkiTanaka/libffm-1.11/ffm-train -l 0.00000001 -k 6 -t 10 -r 0.01 -s 2 -p /Users/IkkiTanaka/Documents/KDDCup/train/ffm/validation.ffm  /Users/IkkiTanaka/Documents/KDDCup/train/ffm/new_train.ffm /Users/IkkiTanaka/Documents/KDDCup/train/ffm/model1.ffm

#validation
!/Users/IkkiTanaka/libffm-1.11/ffm-predict /Users/IkkiTanaka/Documents/KDDCup/train/ffm/validation.ffm /Users/IkkiTanaka/Documents/KDDCup/train/ffm/model1.ffm /Users/IkkiTanaka/Documents/KDDCup/pred/ffm/pred_val.csv

pred_val = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/ffm/pred_val.csv',header=None)

y = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/val_label.csv',header=None)
fpr, tpr, thresholds = roc_curve(y, pd.DataFrame(list(pred_val[0])), pos_label=1)  
print auc(fpr, tpr)

#テスト
!/Users/IkkiTanaka/libffm-1.11/ffm-predict /Users/IkkiTanaka/Documents/KDDCup/train/ffm/new_test.ffm /Users/IkkiTanaka/Documents/KDDCup/train/ffm/model1.ffm /Users/IkkiTanaka/Documents/KDDCup/pred/ffm/prediction1.csv

#all data
!/Users/IkkiTanaka/libffm-1.11/ffm-train -l 0.0001 -k 10 -t 20 -r 0.01 -s 2 /Users/IkkiTanaka/Documents/KDDCup/train/new_train.ffm /Users/IkkiTanaka/Documents/KDDCup/train/model1_all.ffm

!/Users/IkkiTanaka/libffm-1.11/ffm-predict /Users/IkkiTanaka/Documents/KDDCup/train/new_test.ffm /Users/IkkiTanaka/Documents/KDDCup/train/model1_all.ffm /Users/IkkiTanaka/Documents/KDDCup/train/prediction1_all.csv


pred = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/ffm/prediction1.csv',header=None)
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)

test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')

sample.iloc[:,1] = pred[0]

Pred = pd.concat([test['enrollment_id'],pred[0]],axis=1)
Pred1 = Pred.groupby(['enrollment_id']).mean()
#Pred1.columns = 1
#Pred1 = pd.concat([sample.iloc[:,0],Pred1],axis=1)
#Pred1.columns = [0,1]

Pred1.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/ffm/prediction2.csv',header=None)


./perf [options] -files <targets file> /Users/IkkiTanaka/Documents/KDDCup/train/prediction1.txt


