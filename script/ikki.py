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


dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_trainfe10.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_testfe10.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/ikeda/ikeda_labelfe10.csv',header=None)
label = label[0]

#children_train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/ikeda/children_train.csv')
#children_test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/ikeda/children_test.csv')

#dtrain = pd.concat([dtrain,children_train],axis=1)
#dtest = pd.concat([dtest,children_test],axis=1)
#u'00_h_y', u'01_d_y', u'01_h_y', u'01_m_y', u'02_d_y', u'02_h_y', u'02_m_y', u'03_d_y', u'03_h_y', u'04_d_y', u'04_h_y', u'05_d_y', u'05_h_y', u'05_m_y', u'06_d_y', u'06_h_y', u'06_m_y', u'07_d_y', u'07_h_y', u'07_m_y', u'08_d_y', u'08_h_y', u'08_m_y', u'09_d_y', u'09_h_y', u'0_cg_y', u'0_md_y', u'0_w_y', u'10_d_y', u'10_h_y', u'10_m_y', u'11_d_y', u'11_h_y', u'11_m_y', u'12_d_y', u'12_h_y', u'12_m_y', u'13_d_y', u'13_h_y', u'14_d_y', u'14_h_y', u'15_d_y', u'15_h_y', u'16_d_y', u'16_h_y', u'17_d_y', u'17_h_y', u'18_d_y', u'18_h_y', u'19_d_y', u'19_h_y', u'1_md_y', u'1_w_y', u'2013_y', u'2014_y', u'20_d_y', u'20_h_y', u'21_d_y', u'21_h_y', u'22_d_y', u'22_h_y', u'23_d_y', u'23_h_y', u'24_d_y', u'25_d_y', u'26_d_y', u'27_d_y', u'28_d_y', u'29_d_y', u'2_md_y', u'2_w_y', u'30_d_y', u'31_d_y', u'3_md_y', u'3_w_y', u'4_w_y', u'5_md_y', u'5_w_y', u'6_w_y', u'access_y', u'access_unique_y', u'browser_y', u'browser_acc_y', u'browser_pro_y', u'chapter_cg_y', u'combinedopenended_cg_y', u'course_id_count_y', u'discussion_y', u'discussion_unique_y',  u'nagivate_y', u'nagivate_unique_y', u'object_count_all_mean_y', u'object_count_all_std_y', u'object_count_all_sum_y', u'object_count_dtb_mean_y', u'object_count_dtb_std_y', u'object_count_dtb_sum_y', u'page_close_y', u'page_close_unique_y', u'problem_y', u'problem_cg_y', u'problem_unique_y', u'sequential_cg_y', u'server_y', u'server_acc_y', u'server_pro_y', u'sum_count_y', u'username_count_y', u'video_y', u'video_cg_y', u'video_unique_y', u'wiki_y', u'wiki_unique_y'
'''
#日付、moduleの深さのusernameによるgroupbyを削除
#sum
delete_label = [u'object_count_all_mean_y', u'object_count_all_std_y', u'object_count_all_sum_y', u'object_count_dtb_mean_y', u'object_count_dtb_std_y', u'object_count_dtb_sum_y']

for i,j in enumerate(delete_label):
    delete_label[i] = (j + '_user_all')

for i in delete_label:
    try:
        del dtrain[i]
    except:
        print i

for i in delete_label:
    try:
        del dtest[i]
    except:
        print i

#mean
delete_label = [u'object_count_all_mean_y', u'object_count_all_std_y', u'object_count_all_sum_y', u'object_count_dtb_mean_y', u'object_count_dtb_std_y', u'object_count_dtb_sum_y']

for i,j in enumerate(delete_label):
    delete_label[i] = (j + '_user_all_mean')

for i in delete_label:
    try:
        del dtrain[i]
    except:
        print i

for i in delete_label:
    try:
        del dtest[i]
    except:
        print i

#std
delete_label = [u'object_count_all_mean_y', u'object_count_all_std_y', u'object_count_all_sum_y', u'object_count_dtb_mean_y', u'object_count_dtb_std_y', u'object_count_dtb_sum_y']

for i,j in enumerate(delete_label):
    delete_label[i] = (j + '_user_all_std')

for i in delete_label:
    try:
        del dtrain[i]
    except:
        print i

for i in delete_label:
    try:
        del dtest[i]
    except:
        print i


#user_allの時間のみ
#delete_label = ['2013_10_y_user_all','2013_11_y_user_all','2013_12_y_user_all','2014_01_y_user_all','2014_02_y_user_all','2014_05_y_user_all','2014_06_y_user_all','2014_07_y_user_all','2014_08_y_user_all']

#for i,j in enumerate(delete_label):
#    delete_label[i] = (j + '_user_all')

#for i in delete_label:
#    try:
#        del dtrain[i]
#    except:
#        print i

#for i in delete_label:
#    try:
#        del dtest[i]
#    except:
#        print i
'''

'''
#最後にgroupbyしたもの
delete_label = [u'chapter_min_time', u'chapter_max_time', u'chapter_mean_time', u'chapter_std_time', u'chapter_time_max-min', u'chapter_time_max-mean', u'chapter_time_mean-min', u'ratio_of_drop', u'droprate_course', u'ratio_of_drop_uc', u'droprate_uc', u'browser_pct', u'server_pct', u'access_pct', u'discussion_pct', u'nagivate_pct', u'page_close_pct', u'problem_pct', u'video_pct', u'wiki_pct', u'0_cg_pct', u'chapter_cg_pct', u'combinedopenended_cg_pct', u'problem_cg_pct', u'sequential_cg_pct', u'video_cg_pct', u'min_time', u'max_time', u'mean_time', u'std_time', u'len_time', u'max-mean',u'mean-min', u'enr_interval_min', u'enr_interval_max', u'enr_interval_mean', u'enr_interval_std', u'day_count', u'month_count', u'hour_count', u'weekday_count', u'start', u'min-start', u'max-start', u'min-start_chapter', u'max-start_chapter', u'about', u'chapter', u'combinedopenended', u'dictation', u'discussion_y', u'html', u'outlink', u'peergrading', u'problem_y', u'sequential', u'static_tab', u'vertical', u'video_y', u'past_enr', u'past_nondrop_rate', u'num_course_min', u'num_course_max', u'len_from_start', u'len_from_end', u'len_from_start_chapter', u'len_from_end_chapter', u'mean_stay_times', u'browser_per_day', u'server_per_day', u'access_per_day', u'discussion_x_per_day',u'nagivate_per_day', u'page_close_per_day', u'problem_x_per_day', u'video_x_per_day', u'wiki_per_day', u'2013_10', u'2013_11', u'2013_12', u'2014_01', u'2014_02', u'2014_05', u'2014_06', u'2014_07', u'2014_08', u'min_time_count', u'max_time_count', u'mean_time_count', u'min_time_month', u'max_time_month', u'course_count/sum_count', u'username','children_0','children_1','children_sum','children_percent','min_time_problem-min_time','max_time-max_time_problem','max_time_problem-min_time_problem','std_time_problem','mean_time_problem','max_time_problem','min_time_problem','min_time_video-min_time','max_time-max_time_video','max_time_video-min_time_video','std_time_video','mean_time_video','max_time_video','min_time_video']]
'_user_all_new'
'''
#各idの各イベントの時間平均、min,max
train = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/log_test.csv')
train = train.sort('enrollment_id')
test = test.sort('enrollment_id')
train.index = range(0,len(train))
test.index = range(0,len(test))

f_time = lambda x: date(int(x[:4]), int(x[5:7]), int(x[8:10])).toordinal() - 735000
train['time'] = train['time'].apply(f_time)
test['time'] = test['time'].apply(f_time)

#train
aa = train.groupby(['enrollment_id','event']).mean().unstack()['time']
aa.columns += 'mean_time_event_only'
aa = aa.sort_index()
aa.index = range(0,len(aa))
dtrain = pd.concat([dtrain,aa],axis=1)

aa = train.groupby(['enrollment_id','event']).min().unstack()['time']
aa.columns += 'min_time_event_only'
aa = aa.sort_index()
aa.index = range(0,len(aa))
dtrain = pd.concat([dtrain,aa],axis=1)

aa = train.groupby(['enrollment_id','event']).max().unstack()['time']
aa.columns += 'max_time_event_only'
aa = aa.sort_index()
aa.index = range(0,len(aa))
dtrain = pd.concat([dtrain,aa],axis=1)

aa = train.groupby(['enrollment_id','event']).std().unstack()['time']
aa.columns += 'std_time_event_only'
aa = aa.sort_index()
aa.index = range(0,len(aa))
dtrain = pd.concat([dtrain,aa],axis=1)

#test
aa = test.groupby(['enrollment_id','event']).mean().unstack()['time']
aa.columns += 'mean_time_event_only'
aa = aa.sort_index()
aa.index = range(0,len(aa))
dtest = pd.concat([dtest,aa],axis=1)

aa = test.groupby(['enrollment_id','event']).min().unstack()['time']
aa.columns += 'min_time_event_only'
aa = aa.sort_index()
aa.index = range(0,len(aa))
dtest = pd.concat([dtest,aa],axis=1)

aa = test.groupby(['enrollment_id','event']).max().unstack()['time']
aa.columns += 'max_time_event_only'
aa = aa.sort_index()
aa.index = range(0,len(aa))
dtest = pd.concat([dtest,aa],axis=1)

aa = test.groupby(['enrollment_id','event']).std().unstack()['time']
aa.columns += 'std_time_event_only'
aa = aa.sort_index()
aa.index = range(0,len(aa))
dtest = pd.concat([dtest,aa],axis=1)




dtrain_leaf = dtrain
dtest_leaf = dtest
print dtrain_leaf.columns == dtest_leaf.columns
aa = dtrain_leaf.columns

#del dtrain_leaf['sum_count']
f_to_999 = lambda x: 0.00000001 if x==0 else x
#dtrain_leaf = dtrain_leaf.applymap(f_to_999)
#dtest_leaf = dtest_leaf.applymap(f_to_999)


dtrain_leaf = dtrain
dtest_leaf = dtest
print dtrain_leaf.columns == dtest_leaf.columns
aa = dtrain_leaf.columns

dtrain_leaf2 = dtrain_leaf.copy()
dtest_leaf2 = dtest_leaf.copy()

del dtrain_leaf2['past_nondrop_rate_user_all_new'], dtest_leaf2['past_nondrop_rate_user_all_new']
del dtrain_leaf2['past_nondrop_rate_x_user_all_new_mean'], dtest_leaf2['past_nondrop_rate_x_user_all_new_mean']

#username_new_all_mean
print 'dtrain_leaf2.shape: 548?', dtrain_leaf2.shape
print dtrain_leaf.columns == dtest_leaf.columns

dtrain = xgb.DMatrix(dtrain_leaf2, label=label)
dtest = xgb.DMatrix(dtest_leaf2)


pred_name = []
num_model = 300
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
    num_round = int( round(random.uniform(1280, 1280),0))
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
    num_round_ = bst.argmax() + 250#60
    cc = bst.max()
    if cc > 0.8990:
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



param = {'bst:subsample': 0.59, 'eval_metric': 'auc', 'base_score': 0.5, 'num_class': 1, 'bst:eta': 0.01, 'colsample_bytree': 0.84, 'silent': 1, 'bst:gamma': 2.79, 'nthread': 8, 'bst:max_depth': 7, 'objective': 'binary:logistic', 'bst:min_child_weight': 3}
num_round = 1300
