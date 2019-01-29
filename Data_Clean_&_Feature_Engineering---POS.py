import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import itertools
import math
#########################################################
application_train = pd.read_csv('application_train.csv')
application_test = pd.read_csv('application_test.csv')

all = pd.concat([application_train,application_test],axis=0).sort_values(by='SK_ID_CURR')
TARGET = pd.concat([application_train[['SK_ID_CURR','TARGET']],application_test[['SK_ID_CURR']]],axis=0).sort_values(by='SK_ID_CURR')

installments = pd.read_csv('installments_payments.csv').sort_values(by=['SK_ID_CURR','SK_ID_PREV','DAYS_INSTALMENT'])

#########################################################
installments.NUM_INSTALMENT_VERSION.nunique()
installments['DAYS_INSTALLMENT_DIFF'] = installments.DAYS_INSTALMENT - installments.DAYS_ENTRY_PAYMENT
installments['DAYS_INSTALLMENT_DIFF_IND'] = [1 if x>=0 else 0 for x in installments['DAYS_INSTALLMENT_DIFF']]
installments['AMT_INSTALMENT_DIFF'] = installments.AMT_INSTALMENT - installments.AMT_PAYMENT
installments['AMT_INSTALMENT_DIFF_IND'] = [0 if x>=0 else 1 for x in installments['AMT_INSTALMENT_DIFF']]

#############################################################
########################## loan aggregate
temp1 = installments.groupby('SK_ID_PREV').agg({'NUM_INSTALMENT_NUMBER':'max','AMT_INSTALMENT':'sum','DAYS_INSTALMENT':np.ptp,'DAYS_ENTRY_PAYMENT':np.ptp,'AMT_PAYMENT':'sum'})
temp1.columns = ['INSTAL_TOTAL_NUM_INSTALMENT_NUMBER','INSTAL_EXPECTED_AMT','INSTAL_EXPECTED_PAY_DURATION','INSTAL_ACT_PAY_DURATION','INSTAL_ACT_AMT']
temp1['INSTAL_EXPECTED_PAY_PER_DAY'] = temp1.INSTAL_EXPECTED_AMT/temp1.INSTAL_EXPECTED_PAY_DURATION
temp1['INSTAL_EXPECTED_PAY_PER_TIME'] = temp1.INSTAL_EXPECTED_AMT/temp1.INSTAL_TOTAL_NUM_INSTALMENT_NUMBER
temp1['INSTAL_ACT_PAY_PER_DAY'] = temp1.INSTAL_ACT_AMT/temp1.INSTAL_ACT_PAY_DURATION
temp1['INSTAL_EXPECTED_PAY_INTERVAL'] = temp1.INSTAL_EXPECTED_PAY_DURATION/temp1.INSTAL_TOTAL_NUM_INSTALMENT_NUMBER

loan_agg = {'NUM_INSTALMENT_VERSION':['nunique','std'],
            'NUM_INSTALMENT_NUMBER':['max',np.ptp],
            'DAYS_INSTALMENT':['min','max','mean',np.ptp,'std'],
            'AMT_INSTALMENT':['sum','nunique'],
            'AMT_PAYMENT':['sum','nunique'],
            'DAYS_INSTALLMENT_DIFF':['min','max','mean','std'],
            'DAYS_INSTALLMENT_DIFF':['sum','nunique'],
            'AMT_INSTALMENT_DIFF':['min','max','mean','std'],
            'AMT_INSTALMENT_DIFF_IND':['sum','nunique']}

temp2 = installments.groupby('SK_ID_PREV').agg(loan_agg)
temp2.columns = ['INSTAL_'+a+'_'+b for a, b in temp2.columns]
temp3 = temp1.merge(temp2, how='inner', left_index=True, right_index=True)

################################################ customer aggregate
temp4 = installments[['SK_ID_CURR','SK_ID_PREV']].drop_duplicates().merge(temp3.reset_index(),how='left',on='SK_ID_PREV')

cust_agg = {'SK_ID_PREV':'size','INSTAL_TOTAL_NUM_INSTALMENT_NUMBER':'max','INSTAL_EXPECTED_AMT':'sum', 'INSTAL_EXPECTED_PAY_DURATION':'max',
            'INSTAL_ACT_PAY_DURATION':'max','INSTAL_ACT_AMT':'sum','INSTAL_EXPECTED_PAY_PER_DAY':['max','mean','std'],
            'INSTAL_EXPECTED_PAY_PER_TIME':['max','mean','std'], 'INSTAL_ACT_PAY_PER_DAY':['max','mean','std'],
            'INSTAL_EXPECTED_PAY_INTERVAL':['max','std'],'INSTAL_NUM_INSTALMENT_VERSION_nunique':['max','nunique'],'INSTAL_NUM_INSTALMENT_VERSION_std':['max','std'],
            'INSTAL_AMT_INSTALMENT_DIFF_min':'min','INSTAL_AMT_INSTALMENT_DIFF_max':'max', 'INSTAL_AMT_INSTALMENT_DIFF_mean':'mean',
            'INSTAL_AMT_INSTALMENT_DIFF_std':'max', 'INSTAL_DAYS_INSTALMENT_min':'min','INSTAL_DAYS_INSTALMENT_max':'max',
            'INSTAL_DAYS_INSTALMENT_mean':'mean','INSTAL_DAYS_INSTALMENT_ptp':['mean','max'], 'INSTAL_DAYS_INSTALMENT_std':'max',
            'INSTAL_AMT_INSTALMENT_sum':['sum','max'], 'INSTAL_AMT_INSTALMENT_nunique':['max','nunique'],'INSTAL_AMT_PAYMENT_sum':['sum','std'],
            'INSTAL_AMT_PAYMENT_nunique':['max','nunique'],'INSTAL_DAYS_INSTALLMENT_DIFF_sum':['sum','std'],'INSTAL_DAYS_INSTALLMENT_DIFF_nunique':['max','nunique'],
            'INSTAL_NUM_INSTALMENT_NUMBER_max':'max', 'INSTAL_NUM_INSTALMENT_NUMBER_ptp':['max','std'],
            'INSTAL_AMT_INSTALMENT_DIFF_IND_sum':['sum','nunique'],'INSTAL_AMT_INSTALMENT_DIFF_IND_nunique':['max','std']}

temp5 = temp4.groupby('SK_ID_CURR').agg(cust_agg)
temp5.columns = [a+'_'+b for a,b in temp5.columns]

temp5.to_pickle('installments.pkl')
