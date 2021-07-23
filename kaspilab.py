import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


data = pd.read_csv("salary_train.csv")
pred = pd.read_csv("salary_predict.csv")

data_re = data[data['job'] == 'robotics engineer']
data_d = data[data['job'] == 'developer']
data_e = data[data['job'] == 'economist']
data_sd = data[data['job'] == 'senior developer']
data_ds = data[data['job'] == 'data scientist']
data_jd = data[data['job'] == 'junior developer']


# Robotics Engineer

X_re_train = data_re[['robotics', 'programming']]
y_re_train = data_re['salary']

lr_re = LinearRegression()
lr_re.fit(X_re_train, y_re_train)
X_re_predict = pred.loc[pred['job'] == 'robotics engineer'][['robotics', 'programming']]
a = pred.loc[pred['job'] == 'robotics engineer', 'Id'].tolist()
b = np.array(a) - 9000
pred.loc[b, 'salary'] = lr_re.predict(X_re_predict)



# Developer
X_d_train = data_d[['programming']]
y_d_train = data_d['salary']

lr_d = LinearRegression()
lr_d.fit(X_d_train, y_d_train)
X_d_predict = pred.loc[pred['job'] == 'developer'][['programming']]
a = pred.loc[pred['job'] == 'developer', 'Id'].tolist()
b = np.array(a) - 9000
pred.loc[b, 'salary'] = lr_d.predict(X_d_predict)


# Economist
X_e_train = data_e[['economics', 'algebra']]
y_e_train = data_e['salary']

lr_e = LinearRegression()
lr_e.fit(X_e_train, y_e_train)
X_e_predict = pred.loc[pred['job'] == 'economist'][['economics', 'algebra']]
a = pred.loc[pred['job'] == 'economist', 'Id'].tolist()
b = np.array(a) - 9000
pred.loc[b, 'salary'] = lr_e.predict(X_e_predict)



# Senior Developer

X_sd_train = data_sd[['programming']]
y_sd_train = data_sd['salary']

lr_sd = LinearRegression()
lr_sd.fit(X_sd_train, y_sd_train)
X_sd_predict = pred.loc[pred['job'] == 'senior developer'][['programming']]
a = pred.loc[pred['job'] == 'senior developer', 'Id'].tolist()
b = np.array(a) - 9000
pred.loc[b, 'salary'] = lr_sd.predict(X_sd_predict)



# Data Scientist

X_ds_train = data_ds[['programming', 'algebra', 'data science']]
y_ds_train = data_ds['salary']

lr_ds = LinearRegression()
lr_ds.fit(X_ds_train, y_ds_train)
X_ds_predict = pred.loc[pred['job'] == 'data scientist'][['programming', 'algebra', 'data science']]
a = pred.loc[pred['job'] == 'data scientist', 'Id'].tolist()
b = np.array(a) - 9000
pred.loc[b, 'salary'] = lr_ds.predict(X_ds_predict)




# Junior Developer

X_jd_train = data_jd[['programming']]
y_jd_train = data_jd['salary']

lr_jd = LinearRegression()
lr_jd.fit(X_jd_train, y_jd_train)
X_jd_predict = pred.loc[pred['job'] == 'junior developer'][['programming']]
a = pred.loc[pred['job'] == 'junior developer', 'Id'].tolist()
b = np.array(a) - 9000
pred.loc[b, 'salary'] = lr_jd.predict(X_jd_predict)


# writing to file
newpred = pred[['Id', 'salary']]
newpred['salary'] = newpred['salary'].astype(int)
newpred.loc[newpred['salary'] >= 1000000, 'salary'] = 1000000

newpred.to_csv('salary_submition.csv', index = False)