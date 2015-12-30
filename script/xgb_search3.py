from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import sys
sys.path.append('lib')
import xgboost as xgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier

class XGBoostClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, silent=True,
      use_buffer=True,
      num_round=10,
      ntree_limit=0,
      nthread=None,
      booster='gbtree',
      eta=0.1,
      gamma=0.01,
      max_depth=6,
      min_child_weight=1,
      subsample=0.7,
      colsample_bytree=1,      
      l=0,
      alpha=0,
      lambda_bias=0,
      objective='multi:softprob',
      eval_metric='mlogloss',
      seed=0,
      num_class=9,
      max_delta_step=0,
      early_stopping_rounds=None
      ):    
    assert booster in ['gbtree', 'gblinear']
    assert objective in ['reg:linear', 'reg:logistic', 
      'binary:logistic', 'binary:logitraw', 'multi:softmax',
      'multi:softprob', 'rank:pairwise']
    assert eval_metric in [None, 'rmse', 'mlogloss', 'logloss', 'error', 
      'merror',  'auc', 'ndcg', 'map', 'ndcg@n', 'map@n']

    self.silent = silent
    self.use_buffer = use_buffer
    self.num_round = num_round
    self.ntree_limit = ntree_limit
    self.nthread = nthread 
    self.booster = booster
    self.early_stopping_rounds = early_stopping_rounds
    # Parameter for Tree Booster
    self.eta=eta
    self.gamma=gamma
    self.max_depth=max_depth
    self.min_child_weight=min_child_weight
    self.subsample=subsample
    self.colsample_bytree=colsample_bytree
    self.max_delta_step=max_delta_step
    # Parameter for Linear Booster
    self.l=l
    self.alpha=alpha
    self.lambda_bias=lambda_bias
    # Misc
    self.objective=objective
    self.eval_metric=eval_metric
    self.seed=seed
    self.num_class = num_class
    self.classes_ = num_class

  def build_matrix(self, X, opt_y=None):
    if hasattr(X, 'values'): X = X.values
    if opt_y is not None and hasattr(opt_y, 'values'): opt_y = opt_y.values
    return X if hasattr(X, 'handle') else xgb.DMatrix(X, opt_y, missing=np.nan)

  def cv(self, X, y): 
    X = self.build_matrix(X, y)
    param = {
      'silent': 1 if self.silent else 0, 
      'use_buffer': int(self.use_buffer),
      'num_round': self.num_round,
      'ntree_limit': self.ntree_limit,
      'nthread': self.nthread,
      'booster': self.booster,
      'eta': self.eta,
      'gamma': self.gamma,
      'max_depth': self.max_depth,
      'min_child_weight': self.min_child_weight,
      'subsample': self.subsample,
      'colsample_bytree': self.colsample_bytree,
      'max_delta_step': self.max_delta_step,
      'l': self.l,
      'alpha': self.alpha,
      'lambda_bias': self.lambda_bias,
      'objective': self.objective,
      'eval_metric': self.eval_metric,
      'seed': self.seed,
      'num_class': self.num_class,
    }    
    results = xgb.cv(param, X, self.num_round, 5)
    return results

  def fit(self, X, y):    
    X = self.build_matrix(X, y)
    param = {
      'silent': 1 if self.silent else 0, 
      'use_buffer': int(self.use_buffer),
      'num_round': self.num_round,
      'ntree_limit': self.ntree_limit,
      'nthread': self.nthread,
      'booster': self.booster,
      'eta': self.eta,
      'gamma': self.gamma,
      'max_depth': self.max_depth,
      'min_child_weight': self.min_child_weight,
      'subsample': self.subsample,
      'colsample_bytree': self.colsample_bytree,
      'max_delta_step': self.max_delta_step,
      'l': self.l,
      'alpha': self.alpha,
      'lambda_bias': self.lambda_bias,
      'objective': self.objective,
      'eval_metric': self.eval_metric,
      'seed': self.seed          
    }
    if self.classes_ is None:
        self.classes_ = np.unique(y)

    if self.num_class is not None:
      param['num_class']= self.num_class

    watchlist  = [(X,'train')]    
    if self.early_stopping_rounds > 0:
      self.bst = xgb.train(param, X, self.num_round, watchlist, early_stopping_rounds=self.early_stopping_rounds)
    else:
      self.bst = xgb.train(param, X, self.num_round, watchlist)

    return self

  def predict(self, X): 
    X = self.build_matrix(X)
    return self.bst.predict(X)

  def predict_tree(self, X): 
    X = self.build_matrix(X)
    return self.bst.predict(X, pred_leaf = True)

  def predict_proba(self, X):
    #dtest = xgb.DMatrix(X, missing = -999)
    #ypred = self.bst.predict(dtest)
    #return ypred

    X = self.build_matrix(X)
    predictions = self.bst.predict(X)
    if self.objective == 'multi:softprob': return predictions
    return np.vstack([1 - predictions, predictions]).T






param = {"objective" : "binary:logistic",
        "eval_metric" :  'auc',
        #"eval_metric" : "ams@0.15",
        #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
        "num_class" :  1,
        "nthread" :  8,
        "bst:max_depth" :  random.randint(4,5),
        "bst:eta" :  round(random.uniform(.01, .3),2),#step_size
        "bst:gamma" :  round(random.uniform(.0, 3.0),1),#min_loss_reduction
        "bst:min_child_weight" : random.randint(0,8),
        "bst:subsample" :  0.3,#round(random.uniform(.2, 1.0),2),#row_subsample
        "colsample_bytree" :  1,#round(random.uniform(.2, 1.0),2),#column_subsample
        "silent": 1,
        #"scale_pos_weight": scale_pos_weight,
        }
num_round = 400

clf = XGBoostClassifier(nthread=8,booster='gbtree',eta=0.08,gamma=1.0,max_depth=4,min_child_weight=4,subsample=0.9,colsample_bytree=0.9,l=0,alpha=0,lambda_bias=0,objective="binary:logistic",eval_metric='auc',seed=19920407,num_class=1,max_delta_step=0,early_stopping_rounds=None,num_round=1000)


#calibration
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


#clf.cv(dtrain,label[0].values,5)


calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
calibrated_clf.fit(dtrain_sp, label_dtrain[0].values)
pred = calibrated_clf.predict_proba(dval)
print("ROC score", metrics.roc_auc_score(label_dval[0].values, pred[:,1]))

pred = calibrated_clf.predict_proba(dtest)
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)
preds = pd.concat([sample[0],pd.DataFrame(pred[:,1])],axis=1)
preds.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/xgb/sk_GBM2.csv' ,header=None,index=False)



#bagging
clf = XGBoostClassifier(nthread=8,booster='gbtree',eta=0.08,gamma=1.0,max_depth=4,min_child_weight=4,subsample=0.9,colsample_bytree=0.9,l=0,alpha=0,lambda_bias=0,objective="binary:logistic",eval_metric='auc',seed=19920407,num_class=1,max_delta_step=0,early_stopping_rounds=None,num_round=450)

bagging_clf = BaggingClassifier(base_estimator=clf, n_estimators=5, max_samples=.6, max_features=.8, bootstrap=True, bootstrap_features=False, oob_score=True, n_jobs=-1, random_state=19920407, verbose=1)
bagging_clf.fit(dtrain_sp, label_dtrain[0].values)
pred = bagging_clf.predict_proba(dval)
print("ROC score", metrics.roc_auc_score(label_dval[0].values, pred[:,1]))


