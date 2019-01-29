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

all = pd.concat([application_train,application_test],axis=0)
TARGET = pd.concat([application_train[['SK_ID_CURR','TARGET']],application_test[['SK_ID_CURR']]],axis=0)

credit_card_balance = pd.read_csv('credit_card_balance.csv').sort_values(by=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'])
#############
##################################### Feature Engineering
credit_card_balance['UTIL'] = [a/b if b!=0 else 0 for a,b in credit_card_balance[['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']].values]
credit_card_balance[["AMT_DRAWINGS_ATM_CURRENT","AMT_DRAWINGS_CURRENT","AMT_DRAWINGS_OTHER_CURRENT","AMT_DRAWINGS_POS_CURRENT"]].fillna(0,inplace=True)
credit_card_balance['AMT_DRAWINGS_TOTAL'] = credit_card_balance[["AMT_DRAWINGS_ATM_CURRENT","AMT_DRAWINGS_CURRENT","AMT_DRAWINGS_OTHER_CURRENT","AMT_DRAWINGS_POS_CURRENT"]].sum(axis=1)
credit_card_balance["AMT_INST_MIN_REGULARITY"].fillna(0, inplace=True)
credit_card_balance['AMT_PAYMENT_CURRENT'].fillna(credit_card_balance['AMT_PAYMENT_CURRENT'].mean(),inplace=True)
credit_card_balance['AMT_RECEIVABLE_INTEREST'] = 1-credit_card_balance['AMT_RECEIVABLE_PRINCIPAL']/credit_card_balance['AMT_RECIVABLE']
credit_card_balance[["CNT_DRAWINGS_ATM_CURRENT","CNT_DRAWINGS_CURRENT","CNT_DRAWINGS_POS_CURRENT","CNT_INSTALMENT_MATURE_CUM"]].fillna(0, inplace=True)
credit_card_balance['CNT_DRAWINGS_TOTAL'] = credit_card_balance[["CNT_DRAWINGS_ATM_CURRENT","CNT_DRAWINGS_CURRENT","CNT_DRAWINGS_POS_CURRENT","CNT_INSTALMENT_MATURE_CUM"]].sum(axis=1)

credit_card_balance_12 = credit_card_balance.loc[credit_card_balance.MONTHS_BALANCE>=-12]
credit_card_balance_6 = credit_card_balance.loc[credit_card_balance.MONTHS_BALANCE>=-6]
credit_card_balance_3 = credit_card_balance.loc[credit_card_balance.MONTHS_BALANCE>=-3]
credit_card_balance_1 = credit_card_balance.loc[credit_card_balance.MONTHS_BALANCE>=-1]

##################################### loan-level aggregation
loan_agg = {'MONTHS_BALANCE':['min','mean','std',np.ptp],
            'AMT_BALANCE':['min','max','mean','std','sum'],
            'AMT_CREDIT_LIMIT_ACTUAL':['nunique','max','min','std','mean'],
            'UTIL':['min','max','mean','std',np.ptp],
            'AMT_DRAWINGS_ATM_CURRENT':['min','max','mean','std','sum'],
            'AMT_DRAWINGS_CURRENT':['min','max','mean','std','sum'],
            'AMT_DRAWINGS_OTHER_CURRENT':['min','max','mean','std','sum'],
            "AMT_DRAWINGS_POS_CURRENT":['min','max','mean','std','sum'],
            'AMT_DRAWINGS_TOTAL':['min','max','mean','std','sum'],
            'AMT_INST_MIN_REGULARITY':['min','max','mean','std'],
            'AMT_PAYMENT_CURRENT':['min','max','mean','std'],
            'AMT_PAYMENT_TOTAL_CURRENT':['min','max','mean','std'],
            'AMT_RECEIVABLE_PRINCIPAL':['min','max','mean','std','sum'],
            'AMT_TOTAL_RECEIVABLE':['min','max','mean','std','sum'],
            "CNT_DRAWINGS_ATM_CURRENT":['max','sum'],
            "CNT_DRAWINGS_CURRENT":['max','sum'],
            "CNT_DRAWINGS_POS_CURRENT":['max','sum'],
            "CNT_INSTALMENT_MATURE_CUM":['max','sum'],
            "CNT_DRAWINGS_TOTAL":['max','sum'],
            "AMT_RECEIVABLE_INTEREST":['min','max','mean','std']
            }

loan_temp0 = credit_card_balance.groupby('SK_ID_PREV').agg(loan_agg)
loan_temp0.fillna(0,inplace=True)

loan_temp12 = credit_card_balance_12.groupby('SK_ID_PREV').agg(loan_agg)
loan_temp12.fillna(0,inplace=True)

loan_temp6 = credit_card_balance_6.groupby('SK_ID_PREV').agg(loan_agg)
loan_temp6.fillna(0,inplace=True)

loan_temp3 = credit_card_balance_3.groupby('SK_ID_PREV').agg(loan_agg)
loan_temp3.fillna(0,inplace=True)

loan_temp1 = credit_card_balance_1.groupby('SK_ID_PREV').agg(loan_agg)
loan_temp1.fillna(0,inplace=True)
###############################################################################
###################################  customer level aggregation
cust_temp0 = credit_card_balance[['SK_ID_CURR','SK_ID_PREV']].merge(loan_temp0.reset_index(),how='right',on='SK_ID_PREV').drop_duplicates().drop('SK_ID_PREV',axis=1)
cust_temp12 = credit_card_balance_12[['SK_ID_CURR','SK_ID_PREV']].merge(loan_temp12.reset_index(),how='right',on='SK_ID_PREV').drop_duplicates().drop('SK_ID_PREV',axis=1)
cust_temp6 = credit_card_balance_6[['SK_ID_CURR','SK_ID_PREV']].merge(loan_temp6.reset_index(),how='right',on='SK_ID_PREV').drop_duplicates().drop('SK_ID_PREV',axis=1)
cust_temp3 = credit_card_balance_3[['SK_ID_CURR','SK_ID_PREV']].merge(loan_temp3.reset_index(),how='right',on='SK_ID_PREV').drop_duplicates().drop('SK_ID_PREV',axis=1)
cust_temp1 = credit_card_balance_1[['SK_ID_CURR','SK_ID_PREV']].merge(loan_temp1.reset_index(),how='right',on='SK_ID_PREV').drop_duplicates().drop('SK_ID_PREV',axis=1)

cust_agg = {}
for a,b in cust_temp1.drop('SK_ID_CURR',axis=1).columns:
    if b!='ptp':
        cust_agg[a+'_'+b] = b
    elif b=='ptp':
        cust_agg[a+'_'+b] = np.ptp

cust_temp0.set_index('SK_ID_CURR',inplace=True)
cust_temp12.set_index('SK_ID_CURR',inplace=True)
cust_temp6.set_index('SK_ID_CURR',inplace=True)
cust_temp3.set_index('SK_ID_CURR',inplace=True)
cust_temp1.set_index('SK_ID_CURR',inplace=True)

cust_temp0.columns = [a+'_'+b for a,b in cust_temp0.columns]
cust_temp12.columns = [a+'_'+b for a,b in cust_temp12.columns]
cust_temp6.columns = [a+'_'+b for a,b in cust_temp6.columns]
cust_temp3.columns = [a+'_'+b for a,b in cust_temp3.columns]
cust_temp1.columns = [a+'_'+b for a,b in cust_temp1.columns]

cust_temp0 = cust_temp0.reset_index().groupby('SK_ID_CURR').agg(cust_agg)
cust_temp12 = cust_temp12.reset_index().groupby('SK_ID_CURR').agg(cust_agg)
cust_temp6 = cust_temp6.reset_index().groupby('SK_ID_CURR').agg(cust_agg)
cust_temp3 = cust_temp3.reset_index().groupby('SK_ID_CURR').agg(cust_agg)
cust_temp1 = cust_temp1.reset_index().groupby('SK_ID_CURR').agg(cust_agg)

cust_temp0.columns = ['CREDIT_OVERALL_'+c for c in cust_temp0.columns]
cust_temp12.columns = ['CREDIT_PAST12_'+c for c in cust_temp12.columns]
cust_temp6.columns = ['CREDIT_PAST6_'+c for c in cust_temp6.columns]
cust_temp3.columns = ['CREDIT_PAST3_'+c for c in cust_temp3.columns]
cust_temp1.columns = ['CREDIT_PAST1_'+c for c in cust_temp1.columns]


temp = credit_card_balance[['SK_ID_CURR']].merge(cust_temp0,how='left',left_on='SK_ID_CURR',right_index=True)\
                                            .merge(cust_temp12,how='left',left_on='SK_ID_CURR',right_index=True)\
                                            .merge(cust_temp6,how='left',left_on='SK_ID_CURR',right_index=True)\
                                            .merge(cust_temp3,how='left',left_on='SK_ID_CURR',right_index=True)\
                                            .merge(cust_temp1,how='left',left_on='SK_ID_CURR',right_index=True)

################################################################# Additoinal Features
temp.fillna(0, inplace=True)
temp = temp.merge(credit_card_balance.groupby('SK_ID_CURR')['SK_ID_PREV'].agg({'CREDIT_CNT':'count'}),how='left',left_on='SK_ID_CURR',right_index=True)
tempq = pd.get_dummies(credit_card_balance.groupby(['SK_ID_CURR','SK_ID_PREV'])['NAME_CONTRACT_STATUS'].last()).reset_index()
tempq = tempq.rename(columns={'Active':'CREDIT_Active', 'Completed':'CREDIT_Completed', 'Demand':'CREDIT_Demand', 'Signed':'CREDIT_Signed'})
tempq = tempq.groupby('SK_ID_CURR')[['CREDIT_Active', 'CREDIT_Completed', 'CREDIT_Demand', 'CREDIT_Signed']].sum(axis=0)
temp = temp.merge(tempq,how='left',left_on='SK_ID_CURR',right_index=True)

temp.to_pickle('credict.pkl')
