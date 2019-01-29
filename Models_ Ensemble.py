import pandas as pd
import numpy as np

submission_model_1 = pd.read_csv('submission_model_1.csv')
submission_model_2 = pd.read_csv('submission_model_2.csv')
submission_model_3 = pd.read_csv('submission_model_3.csv')
submission_model_4 = pd.read_csv('submission_model_4.csv')
submission_model_5 = pd.read_csv('submission_model_5.csv')
submission_model_6 = pd.read_csv('submission_model_6.csv')
submission_model_7 = pd.read_csv('submission_model_7.csv')

submission =  submission_model_1.merge(submission_model_2,how='inner',on='SK_ID_CURR')
                                .merge(submission_model_3,how='inner',on='SK_ID_CURR')
                                .merge(submission_model_4,how='inner',on='SK_ID_CURR')
                                .merge(submission_model_5,how='inner',on='SK_ID_CURR')
                                .merge(submission_model_6,how='inner',on='SK_ID_CURR')
                                .merge(submission_model_7,how='inner',on='SK_ID_CURR').set_index('SK_ID_CURR')

weights = np.array([0.15]*6+[0.1]).reshape(-1,1)
submission['TARGET'] = submission.valeus.dot(weights).reset_index()
submission.to_csv('final_submission.csv',index=False)
