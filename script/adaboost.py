from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

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

del dtrain['enrollment_id'],dtrain['username'],dtrain['course_id'],
dtrain = pd.concat([b[:len(dtrain)],dtrain],axis=1)
#label = dtrain['drop']
#del dtrain['drop']
#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換


#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test.csv')
#b = categorical(np.array(dtest['course_id']), drop=True)
#b = pd.DataFrame(b)
#dtest['username'] = aaa[0][len(dtrain):]

#dtest['course_id'] = bbb[0][len(dtrain):]



del dtest['enrollment_id'],dtest['username'], dtest['course_id']

xx = b[len(dtrain):]
xx.index = range(0,len(xx))
dtest = pd.concat([xx,dtest],axis=1)
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

#dtrain.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv',index=False)
#dtest.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv',index=False)


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


'''
dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_NN.csv',index=False)
dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/label_NN.csv',index=False)
'''
dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_NN.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_NN.csv')
label = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/label_NN.csv',header=None)
#enr_trainvalを1:2にsplit
n_val = round(len(dtrain)*0.2)
np.random.seed(19920407)
rows = np.random.choice(dtrain.index.values, len(dtrain),replace=False)
dtrain_sp = dtrain.ix[rows[n_val:]]
dval = dtrain.ix[rows[:n_val]]

label_dtrain = label.ix[rows[n_val:]]
label_dval = label.ix[rows[:n_val]]

aa = dtrain.columns

clf0 = RandomForestClassifier(n_estimators=100, random_state=None)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import cross_validation
X = dtrain.values
y = label[0].values
X_test = dtest.values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


skf = cross_validation.StratifiedKFold(y, n_folds=5,random_state=0)
stacked = []
for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf0 = RandomForestClassifier(n_estimators=350,max_features='auto',max_depth=None,min_samples_split=4,min_samples_leaf=2,bootstrap=True,n_jobs=-1,warm_start=False, random_state=19920407)
    clf0.fit(X_train, y_train)
    pred = clf0.predict(X_test)
    print("ROC score", metrics.roc_auc_score(y_test, pred))
    stacked.append(metrics.roc_auc_score(y_test, pred))
np.mean(stacked)





clf = AdaBoostRegressor(base_estimator=clf0, n_estimators=100, learning_rate=.1, loss='square', random_state=None)
clf.fit(dtrain_sp, label_dtrain)
pred = clf.predict(dval)


#calibration
clf = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.05,subsample=.96,max_depth=4,verbose=1,max_features=.96, random_state=None)



calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
calibrated_clf.fit(dtrain_sp, label_dtrain[0].values)
pred = calibrated_clf.predict_proba(dval)
print("ROC score", metrics.roc_auc_score(label_dval[0].values, pred[:,1]))

pred = calibrated_clf.predict_proba(dtest)
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred[:,1])],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/sk_GBM2.csv' ,header=None,index=False)





new_label = a.sort(0).iloc[(a.sort(0)[0]>0.01).values][1].values
clf = GradientBoostingClassifier(n_estimators=400,learning_rate=0.05,subsample=.96,max_depth=4,verbose=1,max_features=.96, random_state=None)
new_dtrain_sp = dtrain_sp[new_label]
new_dval = dval[new_label]
clf.fit(dtrain_sp, label_dtrain[0].values)
pred = clf.predict_proba(dval)
print("ROC score", metrics.roc_auc_score(label_dval[0].values, pred[:,1]))



#GaussianNB
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(dtrain_sp, label_dtrain[0].values)
pred = clf.predict_proba(dval)
print("ROC score", metrics.roc_auc_score(label_dval[0].values, pred[:,1]))


scaler = StandardScaler()
dtrain_sp = scaler.fit_transform(dtrain_sp)
dval = scaler.transform(dval)


from sklearn import svm
clf = svm.SVC(C=1,gamma=0.0,verbose=True,probability=True, max_iter=100, degree=3)
clf.fit(dtrain_sp, label_dtrain[0].values)
pred = clf.predict_proba(dval)
print("ROC score", metrics.roc_auc_score(label_dval[0].values, pred[:,1]))



from sklearn.cluster import KMeans
clf = KMeans(init='k-means++', n_clusters=10, n_init=10)
clf.fit(dtrain_sp, label_dtrain)
pred = clf.predict(dval)


pred1 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/sk_GBM1.csv',header=None)
pred2 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/pred_all21_0.883123_413_5.csv',header=None)
pred3 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/NN/lrelu2ada1_149_483_0.000184_0.000149_0.880200_net0',header=None)
pred4 = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred26.csv',header=None)


final_pred = (pred1[1]+pred2[1]+pred3[1]+pred4[1])/4.0
final_pred = pd.concat([pred1[0],final_pred],axis=1)
final_pred.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Ens3_all.csv',header=None,index=False)

