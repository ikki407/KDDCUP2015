#################################################################
# feature engineering
# LibFFM
#################################################################

import hashlib
import argparse, csv, sys, pickle, collections, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
import matplotlib.pyplot as plt



NR_BINS = 1000000
def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)


fields = [u'username', u'course_id', u'object_fact', u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'0.0_md', u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md', u'about_cg', u'chapter_cg', u'combinedopenended_cg', u'course_cg', u'course_info_cg', u'dictation_cg', u'discussion_cg', u'html_cg',u'outlink_cg', u'peergrading_cg', u'problem_cg', u'sequential_cg', u'static_tab_cg', u'vertical_cg', u'video_cg', u'sum_count', u'time_len']

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
                if field in [u'username', u'course_id']:
                    feats.append('{i}:{v}:{w}'.format(i=i, v=v, w=w))
                    i += 1
                else:
                    feats.append('{i}:{v}:{w}'.format(i=i, v=v, w=row[field]))
                    i += 1
            #print row

            if is_train == True:
                f.write('{0} {1}\n'.format(row['drop'], ' '.join(feats)))
            if is_train == False:
                f.write('{0} {1}\n'.format(0, ' '.join(feats)))

#_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_fm.csv')
#_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv')


#train validation
convert('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_fm.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/new_enr_train.ffm', True)
convert('/Users/IkkiTanaka/Documents/KDDCup/fe_train/enr_validation_fm.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/enr_validation.ffm', True)

#test
#convert('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_ffm/new_enr_test.ffm', False)

#shuffle train
#convert('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train_fm.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/shuffle_enr_train.ffm', True)


#学習
!/Users/IkkiTanaka/libffm-1.11/ffm-train -l 0.0 -k 8 -t 40 -r 0.01 -s 8  -p /Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/enr_validation.ffm  /Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/new_enr_train.ffm /Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/model1.ffm

#validation
!/Users/IkkiTanaka/libffm-1.11/ffm-predict -s 8 /Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/enr_validation.ffm /Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/model1.ffm /Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/pred_val.csv


#ROC
val = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/enr_validation_fm.csv')
#label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)
#label.columns = ['enrollment_id','drop']
#val = val.merge(label,on='enrollment_id',how='left')
#y = val['drop']
pred_val = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/pred_val.csv',header=None)
#pred_val['index'] = val.enrollment_id.values
#pred_val = pred_val.groupby('index').mean()
#y = pd.DataFrame(y)
#y['index'] = val.enrollment_id.values
#y = y.groupby('index').mean()


#y = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/val_enr_label_fm.csv',header=None)
fpr, tpr, thresholds = roc_curve(val['drop'].values, pred_val[0].values, pos_label=1)  
print auc(fpr, tpr)
plt.plot(fpr,tpr)
plt.show()

#テスト
!/Users/IkkiTanaka/libffm-1.11/ffm-predict /Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_ffm/new_enr_test.ffm /Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/model1.ffm /Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/prediction1.csv

#predictionを整える
pred = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/prediction1.csv',header=None)
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)

Pred = pd.concat([sample[0],pred[0]],axis=1)

Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/prediction3.csv',header=None,index=False)

'''
prediction2
0.8036998188037477
-l 0.0 -k 3 -t 160 -r 0.1 -s 8



'''
#./perf [options] -files <targets file> /Users/IkkiTanaka/Documents/KDDCup/train/prediction1.txt

#all data
!/Users/IkkiTanaka/libffm-1.11/ffm-train -l 3.0 -k 10 -t 20 -r 0.01 -s 2 /Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train.ffm /Users/IkkiTanaka/Documents/KDDCup/fe_train/model1_all.ffm

!/Users/IkkiTanaka/libffm-1.11/ffm-predict /Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train.ffm /Users/IkkiTanaka/Documents/KDDCup/fe_train/model1_all.ffm /Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/prediction1_all.csv

#predictionを整える
pred = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/prediction1_all.csv',header=None)
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)

Pred = pd.concat([sample[0],pred[0]],axis=1)

Pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/prediction2_all.csv',header=None,index=False)






pred = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/prediction1_all.csv',header=None)
test_ = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv')

pred['index'] = test_.index
pred2 = pred.groupby('index').mean()
