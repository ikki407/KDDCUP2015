import hashlib
import argparse, csv, sys, pickle, collections, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
from collections import Counter
import matplotlib.pylab as plt

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


#trainデータの中のenrollment_idのヒストグラム
plt.hist(label['enrollment_id'],bins=100)
plt.title('hist of id in train')
plt.ylabel('number of id')
plt.xlabel('id')
plt.show()

#testデータの中のenrollment_idのヒストグラム
plt.hist(enr_test['enrollment_id'],bins=100)
plt.title('hist of id in test')
plt.ylabel('number of id')
plt.xlabel('id')
plt.show()

#truth_trainのdropの図示
plt.plot(label['enrollment_id'],label['drop'])

cnt = Counter()
for i in label['drop']:
    cnt[i]+=1
print '0の数:' ,cnt[0] ,', 1の数:', cnt[1]

