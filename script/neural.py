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

#log transform
f = lambda x: np.log(1+x**2) if x > 1 else x



dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/shuffle_enr_train_depth.csv')
dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/new_enr_test_depth.csv')

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
#train_timelenを読み込んでdtrainに付け加える
train_time.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv',index=False)
test_time.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv',index=False)

#dtrain['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
#dtest['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')

'''

#train_course = dtrain['course_id'] +1
#test_course = dtest['course_id'] +1


#dtrain = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
#dtest = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')

#dtrain = dtrain.iloc[:,39:]
#dtest = dtest.iloc[:,39:]

#0の値を-1にする
#f_to0 = lambda x: -1 if x == 0 else x
#dtrain = dtrain.applymap(f_to0)
#dtest = dtest.applymap(f_to0)


aa = dtrain.columns
#dtrain.iloc[:,39:] = dtrain.iloc[:,39:].applymap(f)#対数変換
#dtest.iloc[:,39:] = dtest.iloc[:,39:].applymap(f)#対数変換

#time幅追加
dtrain['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_timelen.csv')
dtest['time_len'] = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_timelen.csv')


dtrain_leaf = dtrain
dtest_leaf = dtest

dtrain_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_train/NN/train_NN.csv',index=False)
dtest_leaf.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/test_NN.csv',index=False)
label.to_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/label_NN.csv',index=False)


f = lambda x: np.log(1+x) if x > 1 else x

def load_train_data(path,standardize=True):
    df = pd.read_csv(path)
    df = df.applymap(f)
    #df_label = [u'username_count', u'course_id_count', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'sum_count']
    #df = df[df_label]
    #df = df.iloc[:,39:]
    #df = df.applymap(f)
    print df.columns
    #print df.shape
    labels = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/label_NN.csv',header=None)
    labels = labels[0].values
    #df['drop'] = labels
    #print labels
    #del df['drop']#, df['username']
    X = df.values.copy()
    #print X
    #np.random.shuffle(X)
    #X = X[:,:-1]
    #labels = X[:,-1]
    #X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    #print X.columns
    X = X.astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    #print y
    scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    if standardize:
        X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler,standardize=True):
    df = pd.read_csv(path)
    #df_label = [u'username_count', u'course_id_count', u'access', u'discussion', u'nagivate', u'page_close', u'problem', u'video', u'wiki', u'sum_count']
    #df = df[df_label]
    #del df['username']
    #df = df.iloc[:,39:]
    #df = df.applymap(f)
    df = df.applymap(f)
    X = df.values.copy()
    X = X.astype(np.float32)
    ids_ = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/ens7_all.csv',header=None)
    ids = ids_[0].values
    #X = scaler.transform(X)
    if standardize:
        X = scaler.transform(X)
    return X, ids

def make_submission(clf, X_test, ids, encoder, name='my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))

X, y, encoder, scaler = load_train_data('/Users/IkkiTanaka/Documents/KDDCup/fe_train/NN/train_NN.csv')
X_test, ids = load_test_data('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/test_NN.csv', scaler)

rows = np.random.permutation(range(0,len(X)))
X = X[rows,:]
y = y[rows]
#num_classes = len(encoder.classes_)
num_features = X.shape[1]
#y = float32(y)
#f = lambda x: 1 if x==0 else 0
#f = np.vectorize(f)
#y = f(y).astype(np.int32)
y = y.reshape((-1, 1))

layers0 = [('input', InputLayer),
           ('dropout_in', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           #('dense3', DenseLayer),
           #('dropout3', DropoutLayer),
           ('output', DenseLayer)]

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


#X, yをvalidation用に分ける
class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))

learning_rate = theano.shared(np.float32(0.1))
net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout_in_p=0.3,
                 dense0_num_units=256,
                 #dense0_W = GlorotNormal(),
                 dropout0_p=0.5,
                 #dense0_nonlinearity= nonlin.tanh,
                 #dense0_nonlinearity= nonlin.LeakyRectify(0.001),
                 dense1_num_units=512,
                 #dense1_W = GlorotNormal(),
                 #dense1_nonlinearity= nonlin.LeakyRectify(0.001),
                 dropout1_p=0.5,
                 dense2_num_units=1024,
                 #dense2_nonlinearity= nonlin.LeakyRectify(0.001),
                 dropout2_p=0.5,
                 #dense3_num_units=156,
                 #dense3_nonlinearity= nonlin.LeakyRectify(0.001),
                 #dropout3_p=0.6,
                 output_nonlinearity=sigmoid,
                 output_num_units=1,
                 update=adagrad,
                 #update=rmsprop,
                 #update=sgd,
                 #update=nesterov_momentum,
                 update_learning_rate=0.03,
                 #update_momentum=0.9,
                 objective_loss_function = binary_crossentropy,
                 y_tensor_type = T.imatrix,
                 regression=True,
                 #update_learning_rate=learning_rate,
                 #update_momentum=theano.shared(float32(0.9)),
    
                 # on_epoch_finished=[
                 #    AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
                 #    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 # ],
                 #on_epoch_finished=[
                 #       AdjustVariable(learning_rate, target=0, half_life=4),
                 #],
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=32,
                 batch_iterator_train=BatchIterator(batch_size=64),
                 batch_iterator_test=BatchIterator(batch_size=64 )
                 )

net0.fit(X, y)
X_train, X_valid, y_train, y_valid = net0.train_test_split(X, y, net0.eval_size)
probas = net0.predict_proba(X_valid)[:,0]
#probas.loc[(probas[0]<0.3).values,[0]] = 0.1
print("ROC score", metrics.roc_auc_score(y_valid, (probas)))
#yp = DataFrame(net0.predict_proba(X_test),columns=[ u'Class_1', u'Class_2', u'Class_3', u'Class_4', u'Class_5', u'Class_6', u'Class_7', u'Class_8', u'Class_9'])
pred_prob = pd.DataFrame(net0.predict( X_test ))
print pred_prob

sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/predNN1.csv'  ,header=None,index=False)




for i in xrange(0,50):
    layers0 = [('input', InputLayer),
           ('dropout_in', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           #('dense1', DenseLayer),
           #('dropout1', DropoutLayer),
           #('dense2', DenseLayer),
           #('dropout2', DropoutLayer),
           ('output', DenseLayer)]
    random.seed(i*19920407)
    dense_list = [random.randint(10, 150),random.randint(100, 1000),random.randint(100, 1050),random.randint(200, 450)]
    leakness_list = [random.uniform(.0001, .0005),random.uniform(.0001, .0005),random.uniform(.0001, .0005),random.uniform(.0001, .0005)]
    max_iter = random.randint(20,40)
    update_learning_rate = theano.shared(np.float32(random.uniform(.01, .1)))
    dropout_in = round(random.uniform(.1, .5),2)
    batch_size = random.randint(128,256)
    print batch_size
    print update_learning_rate
    print dense_list
    print leakness_list
    print max_iter
    print dropout_in
        
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropout_in_p=dropout_in,
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     #dense1_num_units=dense_list[1],
                     #dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     #dense2_num_units=dense_list[2],
                     #dropout2_p=0.5,
                     #dense2_nonlinearity=sigmoid,
                     #dense3_num_units=dense_list[3],
                     #dropout3_p=0.5,
                     #dense4_num_units=dense_list[4],
                     #dropout4_p=0.5,
                     #dense4_num_units=900,
                     #dropout4_p=0.5,
                     #dropout2_p=0.5,
                     #dense3_num_units=550,
                     #dropout3_p=0.4,
                    #dense4_num_units=512,
                    #dropout4_p=0.3,
                    #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                    #dense0_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[0]),
                    #dense1_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[1]),
                    #dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    #dense3_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[3]),
                    #dense4_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[4]),
                    #dense0_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense1_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense2_nonlinearity= nonlin.LeakyRectify(0.1),
                    output_num_units=1,
                    output_nonlinearity=sigmoid,
                    update=adagrad,
                    #update=adadelta,
                    #update=rmsprop,
                    #update=nesterov_momentum,
                    objective_loss_function = binary_crossentropy,
                    y_tensor_type = T.imatrix,
                    regression=True,

                    update_learning_rate= update_learning_rate,
                    #update_momentum=0.9,
                    #update_learning_rate=theano.shared(float32(0.01)),
                    #update_momentum=theano.shared(float32(0.9)),
    
                    on_epoch_finished=[
                       AdjustVariable('update_learning_rate', start=0.2, stop=0.01),
                    #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    ],
                    #random_state = 407,
                    #l2_costs=0.0001,
                    eval_size=0.2,
                    verbose=1,
                    batch_iterator_train=BatchIterator(batch_size=batch_size),
                    batch_iterator_test=BatchIterator(batch_size=batch_size ),

                    max_epochs=max_iter)
    net0.fit(X, y)
    X_train, X_valid, y_train, y_valid = net0.train_test_split(X, y, net0.eval_size)
    probas = net0.predict_proba(X_valid)[:,0]
    print("ROC score", metrics.roc_auc_score(y_valid, (probas)))
    AUC = metrics.roc_auc_score(y_valid, (probas))
    submission_name = '/Users/IkkiTanaka/Documents/KDDCup/pred/NN/lrelu2ada%i_%i_%i_%f_%f_%f_net0' % (i,dense_list[0],dense_list[1],leakness_list[0],leakness_list[1],AUC)
    pred_prob = pd.DataFrame(net0.predict( X_test ))
    print pred_prob

    sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
    preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
    preds.to_csv(submission_name ,header=None,index=False)
    if AUC > 0.893:

        net1 = NeuralNet(layers=layers0,                     
                     input_shape=(None, num_features),
                     dropout_in_p=dropout_in,
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=0.5,
                     #dense2_nonlinearity=sigmoid,
                     #dense3_num_units=dense_list[3],
                     #dropout3_p=0.5,
                     #dense4_num_units=900,
                     #dropout4_p=0.5,
                     #dropout2_p=0.5,
                     #dense3_num_units=550,
                     #dropout3_p=0.4,
                    #dense4_num_units=512,
                    #dropout4_p=0.3,
                    #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                    dense0_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[0]),
                    dense1_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[1]),
                    dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    #dense3_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[3]),
                    #dense0_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense1_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense2_nonlinearity= nonlin.LeakyRectify(0.1),
                    output_num_units=1,
                    output_nonlinearity=sigmoid,
                    update=adagrad,
                    #update=adadelta,
                    #update=nesterov_momentum,
                    objective_loss_function = binary_crossentropy,
                    y_tensor_type = T.imatrix,
                    regression=True,

                    update_learning_rate=update_learning_rate,
                    #update_momentum=0.9,
                    #update_learning_rate=theano.shared(float32(0.01)),
                    #update_momentum=theano.shared(float32(0.9)),
    
                    #on_epoch_finished=[
                    #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                    #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    #],
                    #random_state = 407,
                    #l2_costs=0.0001,
                    eval_size=0.001,
                    verbose=1,
                    batch_iterator_train=BatchIterator(batch_size=batch_size),
                    batch_iterator_test=BatchIterator(batch_size=batch_size ),

                    max_epochs=max_iter)
        net1.fit(X, y)
        X_train, X_valid, y_train, y_valid = net1.train_test_split(X, y, net1.eval_size)
        probas = net1.predict_proba(X_valid)[:,0]
        print("ROC score", metrics.roc_auc_score(y_valid, (probas)))
        AUC1 = metrics.roc_auc_score(y_valid, (probas))

        submission_name = '/Users/IkkiTanaka/Documents/KDDCup/pred/NN/lrelu2ada%i_%i_%i_%f_%f_%f_net1' % (i,dense_list[0],dense_list[1],leakness_list[0],leakness_list[1],AUC)

        pred_prob = pd.DataFrame(net1.predict( X_test ))
        print pred_prob

        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        preds.to_csv(submission_name ,header=None,index=False)



from sklearn.calibration import CalibratedClassifierCV, calibration_curve
clf = net0
calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
calibrated_clf.fit(X, y)
pred = calibrated_clf.predict_proba(X_test)



# Calibrated with sigmoid calibration
sigmoid = CalibratedClassifierCV(net0, cv=5, method='sigmoid')
pred = sigmoid.predict_proba(X_test)



#keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD,RMSprop
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.regularizers import l1, l2, l1l2
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

X, y, encoder, scaler = load_train_data('/Users/IkkiTanaka/Documents/KDDCup/fe_train/NN/train_NN.csv')
X_test, ids = load_test_data('/Users/IkkiTanaka/Documents/KDDCup/fe_test/NN/test_NN.csv', scaler)
#num_classes = len(encoder.classes_)
num_features = X.shape[1]
#y = y.reshape((-1, 1))

#enr_trainvalを1:2にsplit
n_val = round(len(X)*0.2)
np.random.seed(19920407)
rows = np.random.choice(range(0,len(X)), len(X),replace=False)
dtrain = X[rows[n_val:],:]
dval = X[rows[:n_val],:]

y_dtrain = y[rows[n_val:]]
y_dval = y[rows[:n_val]]

for i in xrange(0,50):
    #random.seed(i*100)
    dense_list = [random.randint(200, 1000),random.randint(200, 1000),random.randint(200, 1000),random.randint(200, 450)]
    max_epoch = 10#random.randint(20, 40)

    model = Sequential()
    model.add(Dense(num_features, dense_list[0], init='glorot_normal'))
    #model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_list[0], dense_list[1], init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_list[1], dense_list[2], init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_list[2], 1, init='glorot_normal'))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
    Optimize = np.random.choice(['RMSprop',sgd],1,replace=False)[0]#,'adadelta','adagrad'
    print num_features, dense_list[0], dense_list[1], dense_list[2], 1
    print max_epoch
    print Optimize
    model.compile(loss='binary_crossentropy', optimizer=Optimize, class_mode="binary")
    model.fit(dtrain, y_dtrain, nb_epoch=max_epoch, batch_size=128,validation_data=(dval,y_dval))
    probas = model.predict_proba(dval)[:,0]
    print("ROC score", metrics.roc_auc_score(y_dval, (probas)))
    pred_prob = pd.DataFrame(model.predict_proba(X_test))
    print pred_prob
    if metrics.roc_auc_score(y_dval, (probas)) > 0.882:
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/NN/predNN_val%i_%i_%i_%i_keras.csv' % (i,dense_list[0],dense_list[1],max_epoch) ,header=None,index=False)

        model = Sequential()
        model.add(Dense(num_features, dense_list[0], init='glorot_normal'))
        #model.add(Dropout(0.3))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(dense_list[0], dense_list[1], init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(dense_list[1], dense_list[2], init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(dense_list[2], 1, init='glorot_normal'))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=Optimize, class_mode="binary")
        model.fit(X, y, nb_epoch=max_epoch, batch_size=128, validation_split=0.0)
        pred_prob = pd.DataFrame(model.predict_proba(X_test))
        print pred_prob
        sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
        preds = pd.concat([sample[0],pd.DataFrame(pred_prob)],axis=1)
        preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/NN/predNN%i_%i_%i_%i_keras.csv' % (i,dense_list[0],dense_list[1],max_epoch) ,header=None,index=False)

'''
#calibration
clf = model
calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
calibrated_clf.fit(X, y)
pred = calibrated_clf.predict_proba(X_test)
'''




p1 =  pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/NN/predNN1_keras.csv',header=None)
p2 =  pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/Pred13.csv',header=None)








model = Sequential()
model.add(Dense(num_features, 524, init='glorot_normal'))
#model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(524, 212, init='glorot_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(212, 1, init='glorot_normal', W_regularizer = l1(.1)))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='RMSprop', class_mode="binary")

model.fit(dtrain, y_dtrain, nb_epoch=30, validation_split=0.2, batch_size=128,validation_data=(dval,y_dval))
probas = model.predict_proba(dval)[:,0]
print("ROC score", metrics.roc_auc_score(y_dval, (probas)))

pred_prob = pd.DataFrame(model.predict_proba(X_test))
print pred_prob





#Long Short-Term Memory in Recurrent Neural Network
X, y, encoder, scaler = load_train_data('/Users/IkkiTanaka/Documents/KDDCup/fe_train/train_NN.csv',standardize=False)
X_test, ids = load_test_data('/Users/IkkiTanaka/Documents/KDDCup/fe_test/test_NN.csv', scaler, standardize=False)
#num_classes = len(encoder.classes_)
num_features = X.shape[1]

model = Sequential()
model.add(Embedding(num_features, 28))
model.add(LSTM(28, 32, activation='tanh', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(32, 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='RMSprop')
model.fit(X, y, nb_epoch=20, validation_split=0.2)
pred = model.predict_proba(X_test)









