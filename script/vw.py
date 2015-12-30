# -*- coding: UTF-8 -*-

###################################
#log_train
#
###################################
import pandas as pd
from datetime import datetime
from csv import DictReader
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np

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
          #if "I" in k: # numerical feature, example: I5
          #  if len(str(v)) > 0: #check for empty values
          #    numerical_features += " %s:%s" % (k,v)
          if k in ['username','course_id','time','source','event','object', u'enrollment_id_count', u'course_id_count', u'source_count', u'event_count', u'object_count', u'username_count']: # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v
			  
	  #Creating the labels		  
      if train: #we care about labels
        if int(labels[labels[0]==int(row['enrollment_id'])][1]) == 1:
          label = 1
        else:
          label = -1 #we set negative label to -1
        outfile.write( "%s | c%s\n" % (label,categorical_features) )
		
      else: #we dont care about labels
        outfile.write( "1 | c%s\n" % (categorical_features) )
      
	  #Reporting progress
      if e % 10000 == 0:
        print("%s\t%s"%(e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))

csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/train/new_train.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/vw/new_train.vw',train=True,labels=label)
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/train/validation.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/vw/validation.vw',train=False)

csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/train/shuffle_train.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/vw/shuffle_train.vw',train=True,labels=label)
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/test/new_test.csv', '/Users/IkkiTanaka/Documents/KDDCup/test/vw/new_test.vw',train=False,labels=label)

cd /Users/IkkiTanaka/Documents/KDDCup
!vw train/new_train.vw -k -c -f train/vw/train.model1.vw --loss_function logistic --passes 1 -l 0.1 -b 30 --nn 50 --holdout_off -q cc

!vw train/vw/validation.vw -t -i train/vw/train.model1.vw -p test/vw/test1.txt

        
import math

outputfile = "/Users/IkkiTanaka/Documents/KDDCup/pred/vw/pred1_val.csv" 

def zygmoid(x):
    return 1 / (1 + math.exp(-x))

with open(outputfile,"wb") as outfile:
    #outfile.write("Id,Predicted\n")
    for line in open("test/vw/test1.txt"):
        row = line.strip().split(" ")
        outfile.write("%f\n"%(zygmoid(float(row[0]))))

#ROC
pred_val = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/vw/pred1_val.csv',header=None)

y = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/train/val_label.csv',header=None)
fpr, tpr, thresholds = roc_curve(y, pd.DataFrame(list(pred_val[0])), pos_label=1)  
print auc(fpr, tpr)



#テスト
!vw test/vw/new_test.vw -t -i train/vw/train.model1.vw -p test/vw/test1.txt

outputfile = "/Users/IkkiTanaka/Documents/KDDCup/pred/vw/pred1.csv" 
with open(outputfile,"wb") as outfile:
    #outfile.write("Id,Predicted\n")
    for line in open("test/vw/test1.txt"):
        row = line.strip().split(" ")
        outfile.write("%f\n"%(zygmoid(float(row[0]))))

pred = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/vw/pred1.csv',header=None)
sample = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/sampleSubmission.csv',header=None)

test = pd.read_csv('/Users/IkkiTanaka/Documents/KDDCup/test/new_test.csv')

sample.iloc[:,1] = pred[0]

Pred = pd.concat([test['enrollment_id'],pred[0]],axis=1)
Pred1 = Pred.groupby(['enrollment_id']).mean()
#Pred1.columns = 1
#Pred1 = pd.concat([sample.iloc[:,0],Pred1],axis=1)
#Pred1.columns = [0,1]

Pred1.to_csv('/Users/IkkiTanaka/Documents/KDDCup/pred/vw/pred1.csv.csv',header=None)

/Users/IkkiTanaka/perf.macosx/perf -ROC -files <targets file> /Users/IkkiTanaka/Documents/KDDCup/train/pred_val.txt

