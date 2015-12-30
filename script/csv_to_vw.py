# -*- coding: UTF-8 -*-

########################################################
# __Author__: Triskelion <info@mlwave.com>             #
# Kaggle competition "Display Advertising Challenge":  #
# http://www.kaggle.com/c/criteo-display-ad-challenge/ #
# Credit: Zygmunt Zając <zygmunt@fastml.com>           #
########################################################
import pandas as pd
from datetime import datetime
from csv import DictReader
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
          if k in ['username','course_id','time','source','event','object']: # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v
			  
	  #Creating the labels		  
      if train: #we care about labels
        if int(labels[labels[0]==int(row['enrollment_id'])][1]) == 1:
          label = 1
        else:
          label = -1 #we set negative label to -1
        outfile.write( "%s '%s  |c%s\n" % (label,row['enrollment_id'],categorical_features) )
		
      else: #we dont care about labels
        outfile.write( "1 '%s |c%s\n" % (row['enrollment_id'],categorical_features) )
      
	  #Reporting progress
      if e % 10000 == 0:
        print("%s\t%s"%(e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))

csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/train/log_train.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/log_train.vw',train=True,labels=label)
csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/train/log_test.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/log_test.vw',train=False,labels=label)

#csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/train/new_train.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/log_train.vw',train=True,labels=label)
#csv_to_vw('/Users/IkkiTanaka/Documents/KDDCup/train/validation.csv', '/Users/IkkiTanaka/Documents/KDDCup/train/log_train.vw',train=True,labels=label)

cd '/Users/IkkiTanaka/Documents/KDDCup/train/'
vw log_train.vw -k -c -f train.model1.vw --loss_function logistic --passes 15 -l 0.1 -b 30 --nn 50 --holdout_period 10 -q cc

vw log_test.vw -t -i train.model1.vw -p test1.txt

        
import math

outputfile = "/Users/IkkiTanaka/Documents/OPT2/VWdata/pred1.csv" 

def zygmoid(x):
    return 1 / (1 + math.exp(-x))

    with open(outputfile,"wb") as outfile:
        #outfile.write("Id,Predicted\n")
        for line in open("/Users/IkkiTanaka/Documents/OPT2/VWdata/test7_3.txt"):
            row = line.strip().split(" ")
            outfile.write("%s,%f\n"%(row[1],zygmoid(float(row[0]))))

#ROC

/Users/IkkiTanaka/perf.macosx/perf -ROC -files <targets file> /Users/IkkiTanaka/Documents/KDDCup/train/pred_val.txt

