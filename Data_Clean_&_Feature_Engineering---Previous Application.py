import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import itertools
import math

#################################### Feature Engineering #########################################################################
def Feature_Engineering(previous_application):
    previous_application['AMT_CREDIT'].loc[previous_application.AMT_CREDIT.isnull()] = previous_application.AMT_CREDIT.mean()
    previous_application['PREV_CREDIT_DIFF'] = previous_application.AMT_CREDIT - previous_application.AMT_APPLICATION
    previous_application['PREV_CREDIT_DIFF_IND'] = [1 if x>=0 else 0 for x in previous_application['PREV_CREDIT_DIFF']]

    AMT_DOWN_PAYMENT_IMPUTE = previous_application.groupby('NAME_CONTRACT_TYPE')['AMT_DOWN_PAYMENT'].mean().fillna(0).to_frame('AMT_DOWN_PAYMENT_IMPUTE')
    previous_application = previous_application.merge(AMT_DOWN_PAYMENT_IMPUTE,how='left',left_on='NAME_CONTRACT_TYPE',right_index=True)
    previous_application['AMT_DOWN_PAYMENT'] = [b if math.isnan(a) else a for a, b in previous_application[['AMT_DOWN_PAYMENT','AMT_DOWN_PAYMENT_IMPUTE']].values]

    previous_application['AMT_GOODS_PRICE'].loc[previous_application.AMT_GOODS_PRICE.isnull()] = previous_application.AMT_GOODS_PRICE.mean()
    previous_application['PREV_CREDIT_VS_GOODS'] = previous_application.AMT_CREDIT/previous_application.AMT_GOODS_PRICE
    previous_application['AMT_ANNUITY'].loc[previous_application.AMT_ANNUITY.isnull()] = previous_application.AMT_ANNUITY.mean()
    previous_application['PREV_CREDIT_VS_ANNUITY'] = previous_application.AMT_CREDIT/previous_application.AMT_ANNUITY
    previous_application['RATE_DOWN_PAYMENT'].loc[previous_application.RATE_DOWN_PAYMENT.isnull()] = previous_application.RATE_DOWN_PAYMENT.mean()
    previous_application['RATE_INTEREST_PRIVILEGED'].loc[previous_application.RATE_INTEREST_PRIVILEGED.isnull()] = previous_application.RATE_INTEREST_PRIVILEGED.mean()
    previous_application['RATE_INTEREST_PRIMARY'].loc[previous_application.RATE_INTEREST_PRIMARY.isnull()] = previous_application.RATE_INTEREST_PRIMARY.mean()

    previous_application['PREV_CREDIT_VS_DAYS'] = previous_application.AMT_CREDIT / previous_application.DAYS_DECISION
    previous_application['NAME_TYPE_SUITE'].loc[previous_application.NAME_TYPE_SUITE.isnull()] = 'Nann'
    previous_application['CNT_PAYMENT'].loc[previous_application.CNT_PAYMENT.isnull()] = previous_application.CNT_PAYMENT.median()
    previous_application['PRODUCT_COMBINATION'].loc[previous_application.PRODUCT_COMBINATION.isnull()] = 'Cash'

    previous_application.fillna(0, inplace=True)

    return previous_application

########################  Deal with Cat_Var
def Cat_var(previous_application):
    previous_cat = previous_application[['SK_ID_PREV','SK_ID_CURR']]
    for col in previous_application.columns:
        if previous_application[col].dtype.name == 'object':
            previous_cat = previous_cat.join(pd.get_dummies(previous_application[col], prefix=col),how='left', on='SK_ID_CURR')
    previous_cat.to_pickle('previous_cat.pkl')
    previous_cat = previous_cat.groupby('SK_ID_CURR')[[c for c in previous_cat.columns if c not in ['SK_ID_CURR','SK_ID_PREV']]].sum()
    previous_cat = previous_application.groupby('SK_ID_CURR')['SK_ID_CURR'].size().to_frame('PREVIOUS_APP_CNT').merge(previous_cat,how='inner',right_index=True,left_index=True)

    return previous_cat
########################  Deal with Num_Var
def Num_var(previous_cat):
    previous_num = previous_application[['SK_ID_PREV','SK_ID_CURR']]
    for col in previous_application.columns:
        if previous_application[col].dtype.name != 'object':
            previous_num[col] = previous_application[col]
    previous_num.to_pickle('previous_num.pkl')

    agg_func = {}
    for col in previous_num.drop(['SK_ID_PREV','SK_ID_CURR'],axis=1).columns:
        agg_func[col] = ['min','mean','max','std','sum']

    previous_num = previous_num.groupby('SK_ID_CURR').agg(agg_func)
    previous_num.columns = [str(a)+'_'+str(b) for a, b in previous_num.columns]

    return previous_num


###########################################################################################
def Feature_selection(previous_cat,previous_cat):
    previous = previous_num.merge(previous_cat,how='inner',left_index=True,right_index=True)
    previous = all[['SK_ID_CURR','TARGET']].merge(previous, how='inner', left_on='SK_ID_CURR', right_index=True)
    previous.fillna(0,inplace=True)
    ready = previous.loc[previous.TARGET.notnull()]
    cat_list = previous_cat.columns
    var_list = previous_num.columns

    _list_ = list(itertools.chain(*[var_list,cat_list]))

    #########################################################################################################
    X_train, X_val, y_train, y_val = train_test_split(ready[_list_], ready.TARGET, test_size=0.33, random_state=99999)

    training_data = lgb.Dataset(X_train, label=y_train, categorical_feature=[])
    validition_data = lgb.Dataset(X_val, label=y_val,  categorical_feature=[])

    num_round = 10000

    param = {'num_leaves': 160,
             'min_data_in_leaf': 149,
             'objective':'binary',
             'max_depth': 7,
             'learning_rate': 0.003,
             "boosting": "gbdt",
             "feature_fraction": 0.7,
             "bagging_freq": 1,
             "bagging_fraction": 0.65 ,
             "bagging_seed": 11,
             "metric": 'auc',
             "lambda_l1": 0.26,
             "random_state": 133,
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
    importance_df

    update_list = importance_df.loc[importance_df.importance>1000].feature.values.tolist()
    update_list.append('SK_ID_CURR')

    previous[update_list].to_pickle('previous.pkl')

    return previous[update_list]


def main():
    application_train = pd.read_csv('application_train.csv')
    application_test = pd.read_csv('application_test.csv')

    all = pd.concat([application_train,application_test],axis=0)
    TARGET = pd.concat([application_train[['SK_ID_CURR','TARGET']],application_test[['SK_ID_CURR']]],axis=0)

    previous_application = pd.read_csv('previous_application.csv')
    previous_application = Feature_Engineering(previous_application)
    previous_cat = Cat_var(previous_application)
    previous_cat = Num_var(previous_application)
    previous = Feature_selection(previous_cat,previous_cat)

if __name__ == '__main__':
        main()
