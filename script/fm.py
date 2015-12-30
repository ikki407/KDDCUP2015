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

dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train_depth.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test_depth.csv')
label = dtrain['drop']
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




#del dtrain['drop']

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

#del dtrain['enrollment_id'],dtrain['username'],dtrain['course_id'],
#dtrain = pd.concat([b[:len(dtrain)],dtrain],axis=1)


#del dtest['enrollment_id'],dtest['username'], dtest['course_id']

#xx = b[len(dtrain):]
#xx.index = range(0,len(xx))
#dtest = pd.concat([xx,dtest],axis=1)
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換

#year, month, day, hour ,weekdayのカウントデータを1にする
f_to1 = lambda x: 1 if x >=1 else 0
label_to1 = [u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w']

dtrain[label_to1] = dtrain[label_to1].applymap(f_to1)
dtest[label_to1] = dtest[label_to1].applymap(f_to1)

label_to1 = [u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki']
dtrain[label_to1] = dtrain[label_to1].applymap(f_to1)
dtest[label_to1] = dtest[label_to1].applymap(f_to1)


#shuffleしてないenr_trainに追加しろ!!
dtrain[['min_time','max_time','mean_time','len_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen_mean.csv')
dtest[['min_time','max_time','mean_time','len_time']] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen_mean.csv')

dtrain['max-mean']=dtrain['max_time']-dtrain['mean_time']
dtrain['mean-min']=dtrain['mean_time']-dtrain['min_time']

dtest['max-mean']=dtest['max_time']-dtest['mean_time']
dtest['mean-min']=dtest['mean_time']-dtest['min_time']


dtrain.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_fm.csv',index=False)
dtest.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv',index=False)


NR_BINS = 1000000
def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)


fields = [u'username', u'course_id', u'drop', u'username_count', u'course_id_count', u'browser', u'server', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'2013', u'2014', u'01_m', u'02_m', u'05_m', u'06_m', u'07_m', u'08_m', u'10_m', u'11_m', u'12_m', u'01_d', u'02_d', u'03_d', u'04_d', u'05_d', u'06_d', u'07_d', u'08_d', u'09_d', u'10_d', u'11_d', u'12_d', u'13_d', u'14_d', u'15_d', u'16_d', u'17_d', u'18_d', u'19_d', u'20_d', u'21_d', u'22_d', u'23_d', u'24_d', u'25_d', u'26_d', u'27_d', u'28_d', u'29_d', u'30_d', u'31_d', u'00_h', u'01_h', u'02_h', u'03_h', u'04_h', u'05_h', u'06_h', u'07_h', u'08_h', u'09_h', u'10_h', u'11_h', u'12_h', u'13_h', u'14_h', u'15_h', u'16_h', u'17_h', u'18_h', u'19_h', u'20_h', u'21_h', u'22_h', u'23_h', u'0_w', u'1_w', u'2_w', u'3_w', u'4_w', u'5_w', u'6_w', u'0.0_md', u'1.0_md', u'2.0_md', u'3.0_md', u'5.0_md', u'0_cg', u'chapter_cg', u'combinedopenended_cg', u'problem_cg', u'sequential_cg', u'video_cg', u'sum_count', u'min_time', u'max_time', u'mean_time', u'len_time', u'max-mean', u'mean-min']
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train_depth.csv')
train = train.iloc[train['module_depth'].notnull().values,:]
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test_depth.csv')
test = test.iloc[train['module_depth'].notnull().values,:]

label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/truth_train.csv',header=None)

def convert(src_path, dst_path, is_train):
    with open(dst_path, 'w') as f:
        for row in csv.DictReader(open(src_path)):
            i = 1
            w = 1
            num = 1
            id = row['enrollment_id']
            feats = []
                
            for field in fields:
                v = hashstr(field+'-'+row[field])
                #print v
                if field in [u'username', u'course_id']:
                    feats.append('{i}:{w}'.format(i=i, w=w))
                    i += 1
                else:
                    feats.append('{i}:{w}'.format(i=i, w=row[field]))
                    i += 1
            #print i
            if is_train == True:
                aaa = train.loc[(train['enrollment_id']==int(id)).values,'object'].drop_duplicates().values
                for k in xrange(0,len(aaa)):
                    feats.append('{i}:{w}'.format(i=i, w=aaa[k]))
                    i+=1
                    #print aaa[k]
            else:
                aaa = test.loc[(test['enrollment_id']==id).values,'object'].drop_duplicates().values
                for k in xrange(0,len(aaa)):
                    feats.append('{i}:{w}'.format(i=i, w=aaa[k]))
                    i+=1
            #print num

            if is_train == True:
                f.write('{0} {1}\n'.format(row['drop'], ' '.join(feats)))
            if is_train == False:
                f.write('{0} {1}\n'.format(0, ' '.join(feats)))
            num += 1
            if num % 10 == 0:
                print num
#_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_fm.csv')
#_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv')


#train validation
convert('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_fm.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/new_enr_train.ffm', True)
#convert('/Users/IkkiTanaka/Documents/KDDCup/fe_train/enr_validation_fm.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/enr_validation.ffm', True)

#test
convert('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_fm.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_ffm/new_enr_test.ffm', False)

#shuffle train
#convert('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train_fm.csv', '/Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/shuffle_enr_train.ffm', True)


#学習
!/Users/IkkiTanaka/libfm-master/bin/libFM -dim '1,1,16' -iter 400 -learn_rate 0.05 -method sgd -init_stdev 2.0 -task c -verbosity 2 -regular '0.01,0,0.01' -train /Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/new_enr_train.ffm -out /Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/pred_val.csv -test /Users/IkkiTanaka/Documents/KDDCup/fe_test/fe_ffm/new_enr_test.ffm


#test
#!/Users/IkkiTanaka/libffm-1.11/ffm-predict -s 8 /Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/enr_validation.ffm /Users/IkkiTanaka/Documents/KDDCup/fe_train/fe_ffm/model1.ffm /Users/IkkiTanaka/Documents/KDDCup/pred/fe_ffm/pred_val.csv


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

