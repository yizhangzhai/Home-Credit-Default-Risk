import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

###############################################################################################################################
application = pd.read_pickle('DNN_application_train_test.pkl')
bureau = pd.read_pickle('DNN_bureau.pkl')
prev = pd.read_pickle('DNN_prev.pkl')
pos = pd.read_pickle('DNN_pos.pkl')
ins = pd.read_pickle('DNN_ins.pkl')
cc = pd.read_pickle('DNN_cc.pkl')


application.drop(["EMERGENCYSTATE_MODE",
            "FONDKAPREMONT_MODE",
            "HOUSETYPE_MODE",
            "NAME_CONTRACT_TYPE",
            "NAME_HOUSING_TYPE",
            "NAME_TYPE_SUITE",
            "WALLSMATERIAL_MODE"],axis=1,inplace=True)

all = application.merge(bureau,how='left',left_on='SK_ID_CURR',right_index=True)
all=all.merge(prev,how='left',left_on='SK_ID_CURR',right_index=True)
all=all.merge(pos,how='left',left_on='SK_ID_CURR',right_index=True)
all=all.merge(ins,how='left',left_on='SK_ID_CURR',right_index=True)
all=all.merge(cc,how='left',left_on='SK_ID_CURR',right_index=True)

all.info()
###################
# embedding_list = [x for x in all.columns if all[x].dtype.name=='O']
embedding_list = ['CODE_GENDER','FLAG_OWN_CAR','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_INCOME_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE','WEEKDAY_APPR_PROCESS_START']
LE = LabelEncoder()
for col in embedding_list:
    all[col].fillna('NaN',inplace=True)
    all[col] =LE.fit_transform(all[col])

ready=all.loc[all.TARGET.notnull().values]
for c in all.columns:
    print('"%s",' %c)

TARGET = all[['SK_ID_CURR','TARGET']]
del application['TARGET']

application = all.loc[:,application.columns]

bureau = all.loc[:,bureau.reset_index().columns]

previous = all.loc[:,prev.reset_index().columns]

POS = all.loc[:,pos.reset_index().columns]

installment = all.loc[:,ins.reset_index().columns]

credit = all.loc[:,cc.reset_index().columns]

##################################################################################################################################
###################################################################

##############################################################################################################  ------------------- MODEL_application
## CODE_GENDER
CAT_CODE_GENDER_INPUT = Input(shape=(1,))
CAT_CODE_GENDER = Embedding(input_dim=application.CODE_GENDER.nunique(),
                            output_dim=round((application.CODE_GENDER.nunique()+1)/2), input_length=1)(CAT_CODE_GENDER_INPUT)
CAT_CODE_GENDER = Reshape(target_shape=(round((application.CODE_GENDER.nunique()+1)/2),))(CAT_CODE_GENDER)
CAT_CODE_GENDER = Dense(5)(CAT_CODE_GENDER)
CAT_CODE_GENDER_OUTPUT = Dropout(0.1)(CAT_CODE_GENDER)

## FLAG_OWN_CAR
CAT_FLAG_OWN_CAR_INPUT = Input(shape=(1,))
CAT_FLAG_OWN_CAR = Embedding(input_dim=application.FLAG_OWN_CAR.nunique(),
                             output_dim=round((application.FLAG_OWN_CAR.nunique()+1)/2), input_length=1)(CAT_FLAG_OWN_CAR_INPUT)
CAT_FLAG_OWN_CAR = Reshape(target_shape=(round((application.FLAG_OWN_CAR.nunique()+1)/2),))(CAT_FLAG_OWN_CAR)
CAT_FLAG_OWN_CAR = Dense(5)(CAT_FLAG_OWN_CAR)
CAT_FLAG_OWN_CAR_OUTPUT = Dropout(0.1)(CAT_FLAG_OWN_CAR)

## NAME_INCOME_TYPE
CAT_NAME_INCOME_TYPE_INPUT = Input(shape=(1,))
CAT_NAME_INCOME_TYPE = Embedding(input_dim=application.NAME_INCOME_TYPE.nunique(),
                             output_dim=round((application.NAME_INCOME_TYPE.nunique()+1)/2), input_length=1)(CAT_NAME_INCOME_TYPE_INPUT)
CAT_NAME_INCOME_TYPE = Reshape(target_shape=(round((application.NAME_INCOME_TYPE.nunique()+1)/2),))(CAT_NAME_INCOME_TYPE)
CAT_NAME_INCOME_TYPE = Dense(5)(CAT_NAME_INCOME_TYPE)
CAT_NAME_INCOME_TYPE_OUTPUT = Dropout(0.1)(CAT_NAME_INCOME_TYPE)

## NAME_EDUCATION_TYPE
CAT_NAME_EDUCATION_TYPE_INPUT = Input(shape=(1,))
CAT_NAME_EDUCATION_TYPE = Embedding(input_dim=application.NAME_EDUCATION_TYPE.nunique(),
                             output_dim=round((application.NAME_EDUCATION_TYPE.nunique()+1)/2), input_length=1)(CAT_NAME_EDUCATION_TYPE_INPUT)
CAT_NAME_EDUCATION_TYPE = Reshape(target_shape=(round((application.NAME_EDUCATION_TYPE.nunique()+1)/2),))(CAT_NAME_EDUCATION_TYPE)
CAT_NAME_EDUCATION_TYPE = Dense(5)(CAT_NAME_EDUCATION_TYPE)
CAT_NAME_EDUCATION_TYPE_OUTPUT = Dropout(0.1)(CAT_NAME_EDUCATION_TYPE)

## NAME_FAMILY_STATUS
CAT_NAME_FAMILY_STATUS_INPUT = Input(shape=(1,))
CAT_NAME_FAMILY_STATUS = Embedding(input_dim=application.NAME_FAMILY_STATUS.nunique(),
                             output_dim=round((application.NAME_FAMILY_STATUS.nunique()+1)/2), input_length=1)(CAT_NAME_FAMILY_STATUS_INPUT)
CAT_NAME_FAMILY_STATUS = Reshape(target_shape=(round((application.NAME_FAMILY_STATUS.nunique()+1)/2),))(CAT_NAME_FAMILY_STATUS)
CAT_NAME_FAMILY_STATUS = Dense(5)(CAT_NAME_FAMILY_STATUS)
CAT_NAME_FAMILY_STATUS_OUTPUT = Dropout(0.1)(CAT_NAME_FAMILY_STATUS)

## OCCUPATION_TYPE
CAT_OCCUPATION_TYPE_INPUT = Input(shape=(1,))
CAT_OCCUPATION_TYPE = Embedding(input_dim=application.OCCUPATION_TYPE.nunique(),
                             output_dim=round((application.OCCUPATION_TYPE.nunique()+1)/2), input_length=1)(CAT_OCCUPATION_TYPE_INPUT)
CAT_OCCUPATION_TYPE = Reshape(target_shape=(round((application.OCCUPATION_TYPE.nunique()+1)/2),))(CAT_OCCUPATION_TYPE)
CAT_OCCUPATION_TYPE = Dense(5)(CAT_OCCUPATION_TYPE)
CAT_OCCUPATION_TYPE_OUTPUT = Dropout(0.1)(CAT_OCCUPATION_TYPE)

## ORGANIZATION_TYPE
CAT_ORGANIZATION_TYPE_INPUT = Input(shape=(1,))
CAT_ORGANIZATION_TYPE = Embedding(input_dim=application.ORGANIZATION_TYPE.nunique(),
                             output_dim=round((application.ORGANIZATION_TYPE.nunique()+1)/2), input_length=1)(CAT_ORGANIZATION_TYPE_INPUT)
CAT_ORGANIZATION_TYPE = Reshape(target_shape=(round((application.ORGANIZATION_TYPE.nunique()+1)/2),))(CAT_ORGANIZATION_TYPE)
CAT_ORGANIZATION_TYPE = Dense(5)(CAT_ORGANIZATION_TYPE)
CAT_ORGANIZATION_TYPE_OUTPUT = Dropout(0.1)(CAT_ORGANIZATION_TYPE)

## WEEKDAY_APPR_PROCESS_START
CAT_WEEKDAY_APPR_PROCESS_START_INPUT = Input(shape=(1,))
CAT_WEEKDAY_APPR_PROCESS_START = Embedding(input_dim=application.WEEKDAY_APPR_PROCESS_START.nunique(),
                             output_dim=round((application.WEEKDAY_APPR_PROCESS_START.nunique()+1)/2), input_length=1)(CAT_WEEKDAY_APPR_PROCESS_START_INPUT)
CAT_WEEKDAY_APPR_PROCESS_START = Reshape(target_shape=(round((application.WEEKDAY_APPR_PROCESS_START.nunique()+1)/2),))(CAT_WEEKDAY_APPR_PROCESS_START)
CAT_WEEKDAY_APPR_PROCESS_START = Dense(5)(CAT_WEEKDAY_APPR_PROCESS_START)
CAT_WEEKDAY_APPR_PROCESS_START_OUTPUT = Dropout(0.1)(CAT_WEEKDAY_APPR_PROCESS_START)

###
MODEL_application_main_INPUT = Input(shape=(application.shape[1]-1-len(embedding_list),))
MDOEL_application_main = Dense(300, kernel_initializer='normal')(MODEL_application_main_INPUT)
MDOEL_application_main = PReLU()(MDOEL_application_main)
MDOEL_application_main = BatchNormalization()(MDOEL_application_main)
MDOEL_application_main = Dropout(0.3)(MDOEL_application_main)
MDOEL_application_main = Dense(120, kernel_initializer='normal')(MDOEL_application_main)
MDOEL_application_main = PReLU()(MDOEL_application_main)
MDOEL_application_main = BatchNormalization()(MDOEL_application_main)
MDOEL_application_main = Dense(50, kernel_initializer='normal')(MDOEL_application_main)
MDOEL_application_main = PReLU()(MDOEL_application_main)
MDOEL_application_main = BatchNormalization()(MDOEL_application_main)
MDOEL_application_main_OUTPUT = Dropout(0.3)(MDOEL_application_main)
###############################################################################################################
##############################################################################################################  ------------------- MODEL_bureau
###
MODEL_bureau_main_INPUT = Input(shape=(bureau.shape[1]-1,))
MDOEL_bureau_main = Dense(80, kernel_initializer='normal')(MODEL_bureau_main_INPUT)
MDOEL_bureau_main = PReLU()(MDOEL_bureau_main)
MDOEL_bureau_main = BatchNormalization()(MDOEL_bureau_main)
MDOEL_bureau_main = Dense(50, kernel_initializer='normal')(MDOEL_bureau_main)
MDOEL_bureau_main = PReLU()(MDOEL_bureau_main)
MDOEL_bureau_main = BatchNormalization()(MDOEL_bureau_main)
MDOEL_bureau_main_OUTPUT = Dropout(0.3)(MDOEL_bureau_main)

##############################################################################################################  ------------------- MODEL_previous
###
MODEL_previous_main_INPUT = Input(shape=(previous.shape[1]-1,))
MODEL_previous_main = Dense(100, kernel_initializer='normal')(MODEL_previous_main_INPUT)
MODEL_previous_main = PReLU()(MODEL_previous_main)
MODEL_previous_main = BatchNormalization()(MODEL_previous_main)
MODEL_previous_main = Dense(50, kernel_initializer='normal')(MODEL_previous_main)
MODEL_previous_main = PReLU()(MODEL_previous_main)
MODEL_previous_main = BatchNormalization()(MODEL_previous_main)
MODEL_previous_main_OUTPUT = Dropout(0.3)(MODEL_previous_main)

##############################################################################################################  ------------------- MODEL_POS
###
MODEL_POS_main_INPUT = Input(shape=(POS.shape[1]-1,))
MODEL_POS_main = Dense(100, kernel_initializer='normal')(MODEL_POS_main_INPUT)
MODEL_POS_main = PReLU()(MODEL_POS_main)
MODEL_POS_main = BatchNormalization()(MODEL_POS_main)
MODEL_POS_main_OUTPUT = Dropout(0.3)(MODEL_POS_main)

##############################################################################################################  ------------------- MODEL_installment
###
MODEL_installment_main_INPUT = Input(shape=(installment.shape[1]-1,))
MODEL_installment_main = Dense(200, kernel_initializer='normal')(MODEL_installment_main_INPUT)
MODEL_installment_main = PReLU()(MODEL_installment_main)
MODEL_installment_main = BatchNormalization()(MODEL_installment_main)
MODEL_installment_main = Dense(50, kernel_initializer='normal')(MODEL_installment_main)
MODEL_installment_main = PReLU()(MODEL_installment_main)
MODEL_installment_main = BatchNormalization()(MODEL_installment_main)
MODEL_installment_main_OUTPUT = Dropout(0.3)(MODEL_installment_main)

##############################################################################################################  ------------------- MODEL_credit
###
MODEL_credit_main_INPUT = Input(shape=(credit.shape[1]-1,))
MODEL_credit_main = Dense(80, kernel_initializer='normal')(MODEL_credit_main_INPUT)
MODEL_credit_main = PReLU()(MODEL_credit_main)
MODEL_credit_main = BatchNormalization()(MODEL_credit_main)
MODEL_credit_main = Dense(40, kernel_initializer='normal')(MODEL_credit_main)
MODEL_credit_main = PReLU()(MODEL_credit_main)
MODEL_credit_main = BatchNormalization()(MODEL_credit_main)
MODEL_credit_main_OUTPUT = Dropout(0.3)(MODEL_credit_main)

##############################################################################################################  ------------------- MODEL_all
#############################################################################################################
MODEL_all_INPUT = Concatenate()([MDOEL_application_main_OUTPUT,
                                        CAT_CODE_GENDER_OUTPUT,
                                        CAT_FLAG_OWN_CAR_OUTPUT,
                                        CAT_NAME_INCOME_TYPE_OUTPUT,
                                        CAT_NAME_EDUCATION_TYPE_OUTPUT,
                                        CAT_NAME_FAMILY_STATUS_OUTPUT,
                                        CAT_OCCUPATION_TYPE_OUTPUT,
                                        CAT_ORGANIZATION_TYPE_OUTPUT,
                                        CAT_WEEKDAY_APPR_PROCESS_START_OUTPUT,
                                        # ###
                                        MDOEL_bureau_main_OUTPUT,
                                        MODEL_previous_main_OUTPUT,
                                        MODEL_POS_main_OUTPUT,
                                        MODEL_installment_main_OUTPUT,
                                        MODEL_credit_main_OUTPUT])

MDOEL_all = Dense(80, kernel_initializer='normal')(MODEL_all_INPUT)
MDOEL_all = PReLU()(MDOEL_all)
MDOEL_all = BatchNormalization()(MDOEL_all)
# MDOEL_all = Dropout(0.3)(MDOEL_all)
# MDOEL_all = Dense(300)(MDOEL_all)
# MDOEL_all = PReLU()(MDOEL_all)
# MDOEL_all = BatchNormalization()(MDOEL_all)
MDOEL_all = Dropout(0.3)(MDOEL_all)
MDOEL_all = Dense(24)(MDOEL_all)
MDOEL_all = PReLU()(MDOEL_all)
MDOEL_all = BatchNormalization()(MDOEL_all)
MDOEL_all = Dropout(0.3)(MDOEL_all)
MDOEL_all = Dense(10, kernel_initializer='normal')(MDOEL_all)
MDOEL_all = PReLU()(MDOEL_all)
MDOEL_all = BatchNormalization()(MDOEL_all)
MDOEL_all = Dropout(0.3)(MDOEL_all)
MDOEL_all_OUTPUT = Dense(1, kernel_initializer='normal', activation='sigmoid')(MDOEL_all)

###################################################################################################################################################
MDOEL_all = Model(inputs=[MODEL_application_main_INPUT,
                                        CAT_CODE_GENDER_INPUT,
                                        CAT_FLAG_OWN_CAR_INPUT,
                                        CAT_NAME_EDUCATION_TYPE_INPUT,
                                        CAT_NAME_FAMILY_STATUS_INPUT,
                                        CAT_NAME_INCOME_TYPE_INPUT,
                                        CAT_OCCUPATION_TYPE_INPUT,
                                        CAT_ORGANIZATION_TYPE_INPUT,
                                        CAT_WEEKDAY_APPR_PROCESS_START_INPUT,
                                        # ###
                                        MODEL_bureau_main_INPUT,
                                        MODEL_previous_main_INPUT,
                                        MODEL_POS_main_INPUT,
                                        MODEL_installment_main_INPUT,
                                        MODEL_credit_main_INPUT],
                                outputs=MDOEL_all_OUTPUT)

MDOEL_all.compile(optimizer='Adam', loss='binary_crossentropy')
MDOEL_all.summary()
# plot_model(MDOEL_all, to_file='MDOEL_all.png', show_shapes=True, show_layer_names=True)

##############################################################################################
temp = application.merge(bureau,how='left',on="SK_ID_CURR")\
                    .merge(previous,how='left',on="SK_ID_CURR")\
                    .merge(POS,how='left',on="SK_ID_CURR")\
                    .merge(installment,how='left',on="SK_ID_CURR")\
                    .merge(credit,how='left',on="SK_ID_CURR")

temp = temp.merge(TARGET,how='inner',on='SK_ID_CURR')
training = temp.loc[temp.TARGET.notnull()].drop(['SK_ID_CURR','TARGET'],axis=1)
testing = temp.loc[~temp.TARGET.notnull()].drop(['SK_ID_CURR','TARGET'],axis=1)

def rank_gauss(x):
    from scipy.special import erfinv
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

for i in training.columns:
    training[i] = rank_gauss(training[i].values)
    testing[i] = rank_gauss(testing[i].values)


X_train, X_val, y_train, y_val = train_test_split(training, temp.TARGET.loc[temp.TARGET.notnull()], test_size=0.33, random_state=10)
###
X_train_concat = [X_train.loc[:,[x for x in application.columns]].drop(embedding_list,axis=1).drop('SK_ID_CURR',axis=1),
                         X_train['CODE_GENDER'],
                         X_train['FLAG_OWN_CAR'],
                         X_train['NAME_EDUCATION_TYPE'],
                         X_train['NAME_FAMILY_STATUS'],
                         X_train['NAME_INCOME_TYPE'],
                         X_train['OCCUPATION_TYPE'],
                         X_train['ORGANIZATION_TYPE'],
                         X_train['WEEKDAY_APPR_PROCESS_START'],
                         X_train.loc[:,[x for x in bureau.columns]].drop('SK_ID_CURR',axis=1),
                         X_train.loc[:,[x for x in previous.columns]].drop('SK_ID_CURR',axis=1),
                         X_train.loc[:,[x for x in POS.columns]].drop('SK_ID_CURR',axis=1),
                         X_train.loc[:,[x for x in installment.columns]].drop('SK_ID_CURR',axis=1),
                         X_train.loc[:,[x for x in credit.columns]].drop('SK_ID_CURR',axis=1)]

X_val_concat = [X_val.loc[:,[x for x in application.columns]].drop(embedding_list,axis=1).drop('SK_ID_CURR',axis=1),
                         X_val['CODE_GENDER'],
                         X_val['FLAG_OWN_CAR'],
                         X_val['NAME_EDUCATION_TYPE'],
                         X_val['NAME_FAMILY_STATUS'],
                         X_val['NAME_INCOME_TYPE'],
                         X_val['OCCUPATION_TYPE'],
                         X_val['ORGANIZATION_TYPE'],
                         X_val['WEEKDAY_APPR_PROCESS_START'],
                         X_val.loc[:,[x for x in bureau.columns]].drop('SK_ID_CURR',axis=1),
                         X_val.loc[:,[x for x in previous.columns]].drop('SK_ID_CURR',axis=1),
                         X_val.loc[:,[x for x in POS.columns]].drop('SK_ID_CURR',axis=1),
                         X_val.loc[:,[x for x in installment.columns]].drop('SK_ID_CURR',axis=1),
                         X_val.loc[:,[x for x in credit.columns]].drop('SK_ID_CURR',axis=1)]



MDOEL_all.fit(X_train_concat,y_train, epochs=10, batch_size=512,validation_data=(X_val_concat,y_val),
                      callbacks=[roc_callback((X_train_concat,y_train),validation_data=(X_val_concat,y_val))])


################################################################################################# entire fitting
 X = [temp.loc[:,[x for x in application.columns]].drop(embedding_list,axis=1).drop(['SK_ID_CURR'],axis=1),
                         temp['CODE_GENDER'],
                         temp['FLAG_OWN_CAR'],
                         temp['NAME_EDUCATION_TYPE'],
                         temp['NAME_FAMILY_STATUS'],
                         temp['NAME_INCOME_TYPE'],
                         temp['OCCUPATION_TYPE'],
                         temp['ORGANIZATION_TYPE'],
                         temp['WEEKDAY_APPR_PROCESS_START'],
                         temp.loc[:,[x for x in bureau.columns]].drop(['SK_ID_CURR'],axis=1),
                         temp.loc[:,[x for x in previous.columns]].drop(['SK_ID_CURR'],axis=1),
                         temp.loc[:,[x for x in POS.columns]].drop(['SK_ID_CURR'],axis=1),
                         temp.loc[:,[x for x in installment.columns]].drop(['SK_ID_CURR'],axis=1),
                         temp.loc[:,[x for x in credit.columns]].drop(['SK_ID_CURR'],axis=1)]



MDOEL_all.fit(X,temp.TARGET, epochs=20, batch_size=512)

from keras.models import load_model
MDOEL_all.save('DNN_all.h5')

#################################################
X_testing_concat = [testing.loc[:,[x for x in application.columns]].drop(embedding_list,axis=1).drop('SK_ID_CURR',axis=1),
                         testing['CODE_GENDER'],
                         testing['FLAG_OWN_CAR'],
                         X_train['NAME_EDUCATION_TYPE'],
                         testing['NAME_FAMILY_STATUS'],
                         testing['NAME_INCOME_TYPE'],
                         testing['OCCUPATION_TYPE'],
                         testing['ORGANIZATION_TYPE'],
                         testing['WEEKDAY_APPR_PROCESS_START'],
                         testing.loc[:,[x for x in bureau.columns]].drop('SK_ID_CURR',axis=1),
                         testing.loc[:,[x for x in previous.columns]].drop('SK_ID_CURR',axis=1),
                         testing.loc[:,[x for x in POS.columns]].drop('SK_ID_CURR',axis=1),
                         testing.loc[:,[x for x in installment.columns]].drop('SK_ID_CURR',axis=1),
                         testing.loc[:,[x for x in credit.columns]].drop('SK_ID_CURR',axis=1)]


res = MDOEL_all.predict(X_testing_concat)
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['TARGET'] = res
sample_submission.to_csv('submission_model_7.csv')
