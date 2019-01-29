##------------------- This code is to process 'Application' dataset
###
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def data_impute(data):
    missing_float_var_impute_list = [x for x in data.drop('TARGET',axis=1).columns if ((sum(data[x].isnull())/data.drop('TARGET',axis=1).shape[0]>0) & (data[x].dtype=='float64'))]
    missing_int_var_impute_list = [x for x in data.drop('TARGET',axis=1).columns if ((sum(data[x].isnull())/data.drop('TARGET',axis=1).shape[0]>0) & (data[x].dtype=='int64'))]

    impute_list = list(itertools.chain(*[missing_float_var_impute_list,missing_int_var_impute_list]))

    for x in impute_list:
        data[x].fillna(data[x].mean(),inplace=True)
    #
    for x in data.drop('TARGET',axis=1).columns:
        if (sum(data[x].isnull())/data.shape[0]>0) & (data[x].dtype=='O'):
            data[x].fillna('Nann',inplace=True)

    return data

def Feature_Engineering(data):
    data['CREDIT_VS_ANNUITY'] = [a/b if b!=0 else 0 for a, b in data[['AMT_CREDIT','AMT_ANNUITY']].values]
    data['CREDIT_VS_GOODS'] = [a/b if b!=0 else 0 for a, b in data[['AMT_CREDIT','AMT_GOODS_PRICE']].values]
    data['CREDIT_VS_INCOME'] = [a/b if b!=0 else 0 for a, b in data[['AMT_CREDIT','AMT_INCOME_TOTAL']].values]
    data['CREDIT_VS_CNT_CHILDREN'] = [a/b if b!=0 else 0 for a, b in data[['AMT_CREDIT','CNT_CHILDREN']].values]
    data['INCOME_VS_CNT_CHILDREN'] = [a/b if b!=0 else 0 for a, b in data[['AMT_INCOME_TOTAL','CNT_CHILDREN']].values]
    data['INCOME_VS_ANNUITY'] = [a/b if b!=0 else 0 for a, b in data[['AMT_INCOME_TOTAL','AMT_ANNUITY']].values]

    data['BIRTH_VS_EMPLOYED'] = [a/b if b!=0 else 0 for a, b in data[['DAYS_BIRTH','DAYS_EMPLOYED']].values]
    data['BIRTH_VS_CAR'] = [(a/365)/b if b!=0 else 0 for a, b in data[['DAYS_BIRTH','OWN_CAR_AGE']].values]
    data['BIRTH_VS_INCOME'] = [(a/365)/b if b!=0 else 0 for a, b in data[['DAYS_BIRTH','AMT_INCOME_TOTAL']].values]
    data['BIRTH_VS_CREDIT'] = [(a/365)/b if b!=0 else 0 for a, b in data[['DAYS_BIRTH','AMT_CREDIT']].values]

    data['OBS_CNT_SOCIAL_CIRCLE_TOTAL'] = data[["OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE","OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE"]].sum(axis=1)
    data['EXT_SOURCE_SUM'] = data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].sum(axis=1)
    data['EXT_SOURCE_MEAN'] = data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
    data['EXT_SOURCE_STD'] = data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
    data['EXT_SOURCE_MULTIP'] = data.EXT_SOURCE_1 * data.EXT_SOURCE_2 * data.EXT_SOURCE_3

    data['CREDIT_VS_AMT_REQ_CREDIT_BUREAU_YEAR'] = [a/b if b!=0 else 0 for a, b in data[['AMT_CREDIT','AMT_REQ_CREDIT_BUREAU_YEAR']].values]

    data['FLAG_DOCUMENT_SUM'] = data[["FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6",
                                    "FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_10","FLAG_DOCUMENT_11",
                                    "FLAG_DOCUMENT_12","FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16",
                                    "FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","FLAG_DOCUMENT_21"]].sum(axis=1)

    data['FLAG_DOCUMENT_KURTOSIS'] = all[["FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6",
                                    "FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_10","FLAG_DOCUMENT_11",
                                    "FLAG_DOCUMENT_12","FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16",
                                    "FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","FLAG_DOCUMENT_21"]].kurtosis(axis=1)

    data.fillna(0, inplace=True)

    return data


def variable_pre_selection(data):

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

    ready=data.loc[TARGET['TARGET'].notnull().values]

    X_train, X_val, y_train, y_val = train_test_split(ready[_list_], TARGET.loc[TARGET.notnull().values], test_size=0.33, random_state=99999)

    training_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_list)
    validition_data = lgb.Dataset(X_val, label=y_val,  categorical_feature=cat_list)

    num_round = 10000

    param = {'num_leaves': 160,
             'min_data_in_leaf': 152,
             'objective':'binary',
             'max_depth': 7,
             'learning_rate': 0.003,
             "boosting": "gbdt",
             "feature_fraction": 0.7,
             "bagging_freq": 1,
             "bagging_fraction": 0.65 ,
             "bagging_seed": 100,
             "metric": 'auc',
             "lambda_l1": 0.26,
             "random_state": 100,
             "verbosity": -1}

    classifier = lgb.train(param,
                            training_data,
                            num_round,
                            valid_sets = [training_data, validition_data],
                            verbose_eval=100,
                            early_stopping_rounds = 200)

    importance_df = pd.DataFrame()
    importance_df["feature"] = _list_
    importance_df["importance"] = classifier.feature_importance('gain')
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    return importance_df, _list_

### main
application_train = pd.read_csv('application_train.csv')
application_test = pd.read_csv('application_test.csv')
all = pd.concat([application_train,application_test],axis=0).sort_values(by=['SK_ID_CURR'])
del all['TARGET']
TARGET = pd.concat([application_train[['SK_ID_CURR','TARGET']],application_test[['SK_ID_CURR']]],axis=0).sort_values(by=['SK_ID_CURR'])

all = data_impute(all)
all = Feature_Engineering(all)
importance_df, _list_ = variable_pre_selection(all) # Evaluate and filter variables
_list_.append('SK_ID_CURR')
all[_list_].to_pickle('application.pkl')
