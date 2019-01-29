import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import itertools
#########################################################
application_train = pd.read_csv('application_train.csv')
application_test = pd.read_csv('application_test.csv')
all = pd.concat([application_train,application_test],axis=0).sort_values(by='SK_ID_CURR')
TARGET = all.TARGET
del all['TARGET']
application = pd.read_pickle('application.pkl').set_index('SK_ID_CURR')
bureau = pd.read_pickle('bureau2_aggregated_to_be_merged_to_application.pkl')
bureau.columns = ['BUR_'+str(c) for c in bureau.columns]
all = application.merge(bureau, how='left', left_index=True, right_index=True)
previous = pd.read_pickle('previous.pkl').set_index('SK_ID_CURR')
previous.columns = ['PREV_'+str(c) for c in previous.columns]
all = all.merge(previous, how='left', left_index=True, right_index=True)
POS = pd.read_pickle('POS.pkl')
all = all.merge(POS, how='left', left_index=True, right_index=True)
installments = pd.read_pickle('installments.pkl')
all = all.merge(installments, how='left', left_index=True, right_index=True)
credict = pd.read_pickle('credict.pkl')
all = all.merge(credict, how='left', left_index=True, right_index=True)
POS_seq_score = pd.read_pickle('POS_seq_score.pkl')
all = all.merge(POS_seq_score, how='left', left_index=True, right_index=True)

####################################################################################################################################################
#                                                                    Training
####################################################################################################################################################
cat_list = ["FLAG_DOCUMENT_3","FLAG_DOCUMENT_6","FLAG_DOCUMENT_8",
            # "FLAG_DOCUMENT_2","FLAG_DOCUMENT_4","FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_5",
            # "FLAG_DOCUMENT_9","FLAG_DOCUMENT_10","FLAG_DOCUMENT_11","FLAG_DOCUMENT_12","FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15",
            # "FLAG_DOCUMENT_16","FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","FLAG_DOCUMENT_21"
            "NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","NAME_TYPE_SUITE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE",
            # "FLAG_MOBIL","FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","FLAG_EMAIL",
            "OCCUPATION_TYPE","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","WEEKDAY_APPR_PROCESS_START","HOUR_APPR_PROCESS_START",
            "REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","LIVE_REGION_NOT_WORK_REGION","REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY","ORGANIZATION_TYPE"]

var_list = ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
            "APARTMENTS_AVG","BASEMENTAREA_AVG","YEARS_BEGINEXPLUATATION_AVG","YEARS_BUILD_AVG","COMMONAREA_AVG","ELEVATORS_AVG","ENTRANCES_AVG",
            "FLOORSMAX_AVG","FLOORSMIN_AVG","LANDAREA_AVG","LIVINGAPARTMENTS_AVG","LIVINGAREA_AVG","NONLIVINGAREA_AVG","APARTMENTS_MODE","BASEMENTAREA_MODE",
            "YEARS_BEGINEXPLUATATION_MODE","YEARS_BUILD_MODE","COMMONAREA_MODE","ENTRANCES_MODE","LANDAREA_MODE","LIVINGAPARTMENTS_MODE","LIVINGAREA_MODE",
            "NONLIVINGAREA_MODE","APARTMENTS_MEDI","BASEMENTAREA_MEDI","YEARS_BEGINEXPLUATATION_MEDI","YEARS_BUILD_MEDI","COMMONAREA_MEDI","ENTRANCES_MEDI",
            "LANDAREA_MEDI","LIVINGAPARTMENTS_MEDI","LIVINGAREA_MEDI",
            #"NONLIVINGAPARTMENTS_AVG",# "FLOORSMAX_MEDI",# "FLOORSMIN_MEDI",# "ELEVATORS_MEDI",# "NONLIVINGAPARTMENTS_MODE",# "FLOORSMIN_MODE",# "FLOORSMAX_MODE",# "ELEVATORS_MODE",
            #"NONLIVINGAPARTMENTS_MEDI",# "NONLIVINGAREA_MEDI",# "FONDKAPREMONT_MODE",# "HOUSETYPE_MODE",# "TOTALAREA_MODE",# "WALLSMATERIAL_MODE",# "EMERGENCYSTATE_MODE"
            "CNT_CHILDREN", "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","REGION_POPULATION_RELATIVE","DAYS_BIRTH", "DAYS_EMPLOYED",
            "DAYS_REGISTRATION","DAYS_ID_PUBLISH","OWN_CAR_AGE","CNT_FAM_MEMBERS",
            "OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE","OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE","DAYS_LAST_PHONE_CHANGE",
            "AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_YEAR",
            ###############
            'CREDIT_VS_ANNUITY', 'CREDIT_VS_GOODS', 'CREDIT_VS_INCOME', 'CREDIT_VS_CNT_CHILDREN', 'INCOME_VS_CNT_CHILDREN', 'BIRTH_VS_EMPLOYED',
            'BIRTH_VS_CAR', 'OBS_CNT_SOCIAL_CIRCLE_TOTAL', 'EXT_SOURCE_SUM', 'EXT_SOURCE_MEAN', 'EXT_SOURCE_STD', 'EXT_SOURCE_MULTIP',
            'INCOME_VS_ANNUITY','FLAG_DOCUMENT_SUM','FLAG_DOCUMENT_KURTOSIS','BIRTH_VS_INCOME','BIRTH_VS_CREDIT','CREDIT_VS_AMT_REQ_CREDIT_BUREAU_YEAR']

_list_ = list(itertools.chain(*[var_list,cat_list]))
_list_ = list(itertools.chain(*[_list_,bureau.columns.tolist()]))
_list_ = list(itertools.chain(*[_list_,previous.columns.tolist()]))
_list_ = list(itertools.chain(*[_list_,POS.columns.tolist()]))
_list_ = list(itertools.chain(*[_list_,installments.columns.tolist()]))
_list_ = list(itertools.chain(*[_list_,credict.columns.tolist()]))
_list_ = list(itertools.chain(*[_list_,POS_seq_score.columns.tolist()]))

#########################################################################################################################################

ready=all.loc[TARGET.notnull().values]

training_data = lgb.Dataset(ready[_list_], label=TARGET.loc[TARGET.notnull().values], categorical_feature=cat_list)

num_round = 5000

param = {'num_leaves': 5000,
         'min_data_in_leaf': 30,
         'objective':'binary',
         'max_depth': 8,
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.85 ,
         "bagging_seed": 60,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "random_state": 2,
         "verbosity": -1}

classifier = lgb.train(param,
                        training_data,
                        num_round,
                        valid_sets = [training_data, training_data],
                        verbose_eval=100,
                        early_stopping_rounds = 200)

####### submission
testing = all.loc[TARGET.isnull().values]
submission = pd.DataFrame()
submission['SK_ID_CURR'] = testing.index

res = classifier.predict(testing[_list_])
submission['TARGET'] = res
sample_submission = pd.read_csv('sample_submission.csv')
submission = sample_submission[['SK_ID_CURR']].merge(submission, how='left', on='SK_ID_CURR')
submission.to_csv('submission_model_5.csv', index=False)
