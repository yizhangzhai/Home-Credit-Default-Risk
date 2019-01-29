##------------------- This code is to process 'Bureau' and 'Bureau_balance' datasets
###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import roc_auc_score
from keras.models import Model, Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences

def bureau_balance_feature_engineering(data):
    new_balance = data.merge(bureau[['SK_ID_BUREAU','CREDIT_ACTIVE']], how='inner', on='SK_ID_BUREAU')
    temp0 = new_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].agg({'MONTHS_BALANCE_MIN':'min','MONTHS_BALANCE_CNT':'size'})
    temp1 = new_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max().reset_index()
    temp1 = new_balance.merge(temp1, how='inner', on=['SK_ID_BUREAU','MONTHS_BALANCE']).rename(columns={'STATUS':'STATUS_prior_0'})
    temp2 = (new_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()-2).reset_index()
    temp2 = new_balance.merge(temp2, how='inner', on=['SK_ID_BUREAU','MONTHS_BALANCE']).rename(columns={'STATUS':'STATUS_prior_3'})
    temp3 = (new_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()-5).reset_index()
    temp3 = new_balance.merge(temp3, how='inner', on=['SK_ID_BUREAU','MONTHS_BALANCE']).rename(columns={'STATUS':'STATUS_prior_6'})
    temp4 = (new_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()-11).reset_index()
    temp4 = new_balance.merge(temp4, how='inner', on=['SK_ID_BUREAU','MONTHS_BALANCE']).rename(columns={'STATUS':'STATUS_prior_12'})

    temp5 = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].apply(lambda x: [int(y) for y in x.tolist() if y.isdigit()]).reset_index()
    temp5['DLNQ_SUM'] = [sum(x) for x in temp5.STATUS.values]


    temp = temp0.reset_index().merge(temp1.drop(['MONTHS_BALANCE'],axis=1),how='left',on='SK_ID_BUREAU')\
                                .merge(temp2.drop(['MONTHS_BALANCE'],axis=1),how='left',on='SK_ID_BUREAU')\
                                .merge(temp3.drop(['MONTHS_BALANCE'],axis=1),how='left',on='SK_ID_BUREAU')\
                                .merge(temp4.drop(['MONTHS_BALANCE'],axis=1),how='left',on='SK_ID_BUREAU')\
                                .merge(temp5[['SK_ID_BUREAU','DLNQ_SUM']],how='left',on='SK_ID_BUREAU')

    ### Embedding STATUS
    new_balance['STATUS'] = new_balance.STATUS.map({'X':'7', '0':'0', 'C':'6', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5'})
    temp_embed_X = new_balance.groupby('SK_ID_BUREAU')['STATUS'].apply(lambda x: x.tolist())
    temp_embed_X = pad_sequences(temp_embed_X, maxlen=40)
    temp_embed_y = new_balance.groupby('SK_ID_BUREAU')['CREDIT_ACTIVE'].last()
    temp_embed_y = pd.get_dummies(temp_embed_y.to_frame())

    input = Input(shape=(40,))
    embedding = Embedding(input_dim=8,output_dim=1)(input)
    embedding = Flatten()(embedding)
    output = Dense(4,activation='softmax')(embedding)
    embed_STATUS = Model(inputs=input, outputs=output)
    embed_STATUS.compile(optimizer='Adam',loss='categorical_crossentropy')
    embed_STATUS.summary()
    embed_STATUS.fit(temp_embed_X,temp_embed_y,epochs=3,batch_size=128)

    dict = {str(i):v[0] for i, v in enumerate(embed_STATUS.layers[1].get_weights()[0])}
    temp['STATUS_prior_0'] = temp.STATUS_prior_0.map({'X':'7', '0':'0', 'C':'6', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5'}).map(dict)
    temp['STATUS_prior_3'] = temp.STATUS_prior_3.map({'X':'7', '0':'0', 'C':'6', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5'}).map(dict)
    temp['STATUS_prior_6'] = temp.STATUS_prior_6.map({'X':'7', '0':'0', 'C':'6', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5'}).map(dict)
    temp['STATUS_prior_12'] = temp.STATUS_prior_12.map({'X':'7', '0':'0', 'C':'6', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5'}).map(dict)
    temp['STATUS_prior_0'] = temp['STATUS_prior_0']/1
    temp['STATUS_prior_3'] = temp['STATUS_prior_3']/3
    temp['STATUS_prior_6'] = temp['STATUS_prior_3']/6
    temp['STATUS_prior_12'] = temp['STATUS_prior_3']/12
    temp.fillna(0, inplace=True)

    return temp

def bureau_feature_engineering(data,temp):
    bureau = data.sort_values(by=['SK_ID_CURR','DAYS_CREDIT'])
    bureau['DLNQ_NUCKET'] = [(x//30+1) for x in bureau.CREDIT_DAY_OVERDUE]
    bureau['Total_CREDIT_LOAN_AMT'] = bureau['AMT_CREDIT_SUM'] + bureau['AMT_CREDIT_SUM_DEBT']
    bureau['CREDIT_UTIL'] =  [a/b if b>0 else 0 for a, b in bureau[['AMT_CREDIT_SUM','AMT_CREDIT_SUM_LIMIT']].values]
    bureau = pd.get_dummies(bureau,columns=['CREDIT_ACTIVE'])
    bureau = pd.get_dummies(bureau,columns=['CREDIT_TYPE'])
    bureau = pd.get_dummies(bureau,columns=['CREDIT_CURRENCY'])
    bureau = bureau.merge(temp,how='left',on='SK_ID_BUREAU')
    bureau.fillna(0, inplace=True)

    return bureau

#######################################  Aggregated data to be merged into application dataset
def aggregate_bureau(data):
        bureau2 = data
        bureau2_temp1 = bureau2.groupby('SK_ID_CURR').apply(lambda x: x.AMT_CREDIT_SUM.sum()/(x.AMT_CREDIT_SUM_DEBT.sum()+1))
        bureau2_temp2 = bureau2.groupby('SK_ID_CURR').apply(lambda x: x.AMT_CREDIT_SUM.sum() + x.AMT_CREDIT_SUM_DEBT.sum())
        bureau2_temp3 = bureau2.groupby('SK_ID_CURR').apply(lambda x: x.AMT_CREDIT_SUM_OVERDUE.sum()/(x.AMT_CREDIT_SUM.sum()+1))
        bureau2_temp4 = bureau2.groupby('SK_ID_CURR').apply(lambda x: x.AMT_CREDIT_SUM.sum()/(x.AMT_CREDIT_SUM_LIMIT.sum()+1))
        bureau2_temp5 = bureau2.groupby('SK_ID_CURR').apply(lambda x: x.AMT_CREDIT_SUM.sum()/(x.AMT_ANNUITY.sum()+1))

        agg_func = {'DAYS_CREDIT':['min','mean','max','std',np.ptp],
                    'CREDIT_DAY_OVERDUE':['max','sum'],
                    'DAYS_CREDIT_ENDDATE':['min','max','mean','std'],
                    'DAYS_ENDDATE_FACT':['min','max','mean','std'],
                    'AMT_CREDIT_SUM_OVERDUE':['min','max','mean'],
                    'CNT_CREDIT_PROLONG':'sum',
                    'AMT_CREDIT_SUM':['min','max','mean','std'],
                    'AMT_CREDIT_SUM_DEBT':['min','max','mean','std'],
                    'AMT_CREDIT_SUM_LIMIT':['max','sum','mean'],
                    'AMT_ANNUITY':['max','sum'],
                    'DLNQ_NUCKET':['nunique','min','max','mean'],
                    'Total_CREDIT_LOAN_AMT':['min','max','mean','std'],
                    'CREDIT_UTIL':['max','mean','std'],
                    'CREDIT_ACTIVE_Active':'sum',
                    'CREDIT_ACTIVE_Bad debt':'sum',
                    'CREDIT_ACTIVE_Closed':'sum',
                    'CREDIT_ACTIVE_Sold':'sum',
                    'CREDIT_TYPE_Another type of loan':'sum',
                    'CREDIT_TYPE_Car loan':'sum',
                    'CREDIT_TYPE_Cash loan (non-earmarked)':'sum',
                    'CREDIT_TYPE_Consumer credit':'sum',
                    'CREDIT_TYPE_Credit card':'sum',
                    'CREDIT_TYPE_Interbank credit':'sum',
                    'CREDIT_TYPE_Loan for business development':'sum',
                    'CREDIT_TYPE_Loan for purchase of shares (margin lending)':'sum',
                    'CREDIT_TYPE_Loan for the purchase of equipment':'sum',
                    'CREDIT_TYPE_Loan for working capital replenishment':'sum',
                    'CREDIT_TYPE_Microloan':'sum',
                    'CREDIT_TYPE_Mobile operator loan':'sum',
                    'CREDIT_TYPE_Mortgage':'sum',
                    'CREDIT_TYPE_Real estate loan':'sum',
                    'CREDIT_TYPE_Unknown type of loan':'sum',
                    'CREDIT_CURRENCY_currency 1':'sum',
                    'CREDIT_CURRENCY_currency 2':'sum',
                    'CREDIT_CURRENCY_currency 3':'sum',
                    'CREDIT_CURRENCY_currency 4':'sum',
                    'MONTHS_BALANCE_MIN':['min','max','mean','std'],
                    'MONTHS_BALANCE_CNT':['min','max','mean','std'],
                    'STATUS_prior_0':['mean','sum'],
                    'STATUS_prior_3':['mean','sum'],
                    'STATUS_prior_6':['mean','sum'],
                    'STATUS_prior_12':['mean','sum'],
                    'DLNQ_SUM':['max','sum']
                    }

        bureau2_temp = bureau2.groupby('SK_ID_CURR').agg(agg_func)
        bureau2_temp.columns = [str(a)+'_'+str(b) for a,b in bureau2_temp.columns]

        bureau2_temp1 = bureau2_temp1.to_frame('BUREAU_CREDIT_OVER_DEBT')
        bureau2_temp2 = bureau2_temp2.to_frame('BUREAU_TOTAL_CREDIT')
        bureau2_temp3 = bureau2_temp3.to_frame('BUREAU_OVERDUE_RATIO')
        bureau2_temp4 = bureau2_temp4.to_frame('BUREAU_CREDIT_UTIL')
        bureau2_temp5 = bureau2_temp5.to_frame('BUREAU_CREDIT_OVER_ANNUITY')

        bureau2_temp = bureau2_temp.merge(bureau2_temp1, how='left', left_index=True, right_index=True)\
                                    .merge(bureau2_temp2, how='left', left_index=True, right_index=True)\
                                    .merge(bureau2_temp3, how='left', left_index=True, right_index=True)\
                                    .merge(bureau2_temp4, how='left', left_index=True, right_index=True)\
                                    .merge(bureau2_temp5, how='left', left_index=True, right_index=True)

        bureau2_temp.to_pickle('bureau2_aggregated_to_be_merged_to_application.pkl')

####################################### Turn bureau_loan into sequence
def bureau_turn_to_seq(data):
    bureau_loan_seq = all[['SK_ID_CURR']].merge(data,how='left',on='SK_ID_CURR')
    bureau_loan_seq = bureau_loan_seq.sort_values(by=['SK_ID_CURR','DAYS_CREDIT'])
    bureau_loan_seq = bureau_loan_seq.set_index('SK_ID_BUREAU')
    bureau_loan_seq.fillna(0, inplace=True)

    seq = np.zeros((bureau_loan_seq.SK_ID_CURR.nunique(),20,bureau_loan_seq.shape[1]-1))
    id = []
    i = 0
    for d, sub in bureau_loan_seq.groupby('SK_ID_CURR'):
        id.append(d)
        for j, s in enumerate(sub.drop(['SK_ID_CURR'],axis=1).values):
            if j > 8:
                continue
            else:
                seq[i,-j,:] = s
        i +=1
    return id, bureau_loan_seq


### main
application_train = pd.read_csv('application_train.csv')
application_test = pd.read_csv('application_test.csv')

bureau = pd.read_csv('bureau.csv')
bureau_balance = pd.read_csv('bureau_balance.csv')

bureau = bureau.sort_values(by=['SK_ID_CURR','SK_ID_BUREAU'])
bureau_balance = bureau_balance.sort_values(by=['SK_ID_BUREAU','MONTHS_BALANCE'])

all = pd.concat([application_train,application_test],axis=0)
temp = bureau_balance_feature_engineering(all)
bureau_ = bureau_feature_engineering(all,temp)
bureau2_temp = aggregate_bureau(bureau_)
id, bureau_loan_seq = bureau_turn_to_seq(bureau_)
