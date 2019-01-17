# Home-Credit-Default-Risk
Predict loan borrowers' repayment abilities

![alt text](https://miro.medium.com/max/1276/1*H-Y1yWuKODyqIFD__0ajkg.png)

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.
Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. 
In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, 
maturity, and repayment calendar that will empower their clients to be successful.

This project contains multiple relational datasets from difference sources, such as applicants' profile as of application day, applicants' bureau information and their previous loan records. At the bottom are the details about the datasets.

This is a binary classification project, where 8% borrowers could not afford their loans. And the metrics is ROC, which is a standard for binary and imbalance classification problem.

Per modeling, based on my working experience in building risk models in bank, I have done series of feature engineering related to borrowers application financial status and their belonging to some specific segments. The most important ones are those corresponding to borrowers' utilization, the interest rates, dlinquent status, etc..; I also focus on precise aggregation on those transactional data to capture the behavior patterns of borrowers, like how they managed their single/multiple loans and how they did repayments.

In terms of models, my champion models rely on lightgbm and I also built deep learning models (DNN, Entity Embedding, CNN/LSTM) to solve this complicated problem.


---------------------------------------------------------------------------------------------------------------------------------------
application_{train|test}.csv ---
This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
Static data for all applications. One row represents one loan in our data sample.

bureau.csv ---
All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.

bureau_balance.csv ---
Monthly balances of previous credits in Credit Bureau.
This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.

POS_CASH_balance.csv ---
Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

credit_card_balance.csv ---
Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

previous_application.csv ---
All previous applications for Home Credit loans of clients who have loans in our sample.
There is one row for each previous application related to loans in our data sample.

installments_payments.csv ---
Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
There is a) one row for every payment that was made plus b) one row each for missed payment.
One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.
HomeCredit_columns_description.csv
