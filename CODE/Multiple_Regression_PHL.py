#Random Forest Analysis was leveraged to perform data clean up and formating. Author: wsc72
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score,mean_squared_error
from functools import partial
import time
import os,sys,pickle
import seaborn as sns
from sklearn import metrics
from datetime import timedelta
from datetime import datetime

df_CPI = pd.read_excel('BLS CPI U DC Metro 1980-2020.xlsx', header = 11, usecols = "A:M")
df_CPI.index = df_CPI["Year"]
df_CPI_months=df_CPI.drop(columns=["Year"])
df_CPI_months=df_CPI_months.stack().reset_index()
df_CPI_months.columns=["Year","Month","CPI"]
df_CPI_months["Year"]=df_CPI_months.Year.astype(str)

df_unemployment = pd.read_excel('BLS DC Unemployment 1990-2020.xlsx', header = 10, usecols = "A:M")
df_unemployment.index = df_unemployment["Year"]
df_unemployment_months=df_unemployment.drop(columns=["Year"])
df_unemployment_months=df_unemployment_months.stack().reset_index()
df_unemployment_months.columns=["Year","Month","unemployment"]
df_unemployment_months["Year"]=df_unemployment_months.Year.astype(str) 

df_interest_rate=pd.read_excel("Federal Reserve Interest Rates Data.xlsx",header=0,usecols="A:L")
df_interest_rate["year"]=df_interest_rate.Month.apply(lambda x:x.split("-")[0])
df_interest_rate["Month"]=df_interest_rate.Month.apply(lambda x: datetime.strptime(x.split("-")[-1], "%m").strftime("%b"))
df_interest_rate=df_interest_rate[['year','Month','Federal Funds Rate','PRIME Rate']]
df_interest_rate.columns=['Year','Month','Federal-Funds-Rate','PRIME-Rate']

df_merge=pd.merge(df_CPI_months,df_unemployment_months,left_on=["Year","Month"],right_on=["Year","Month"],how="outer")
df_merge_final=pd.merge(df_merge,df_interest_rate,left_on=["Year","Month"],right_on=["Year","Month"],how="outer")
df_merge_final=df_merge_final.sort_values(by=["Year","Month"])

data_df_main=pd.read_excel('PHL_assessment_clean.xls', header = 0, usecols = "A:Y")
data_df_main["Year"]=data_df_main.Sale_Date.apply(lambda x:x.year)
data_df_main["Sale_Month"]=data_df_main.Sale_Date.apply(lambda x: datetime.strptime(str(x.month), "%m").strftime("%b"))
data_df_main["Sale_Year"]=data_df_main.Year.astype(str)

final_df=pd.merge(data_df_main,df_merge_final,left_on=["Sale_Year","Sale_Month"],right_on=["Year","Month"],how="outer")

y_columns=["Sale_Price"]
dummy_columns=['Building_Category', 'Air_Conditioning']

num_columns=["Sale_Year",'Bedroom_Count', 'Full_Bathroom_Count', 'Total_Room_Count', 'Basement_Room_Count', 'Fireplaces_Count', 'Basement_Garage_Parking_Space_Count','Physical_Condition', 'Non_Basement_Area','Number_of_Stories','Lot_Size']
join_columns=['CPI', 'unemployment', 'Federal-Funds-Rate','PRIME-Rate']

filtered_df=final_df[y_columns+dummy_columns+num_columns+join_columns]

filtered_df=filtered_df[filtered_df.apply(lambda row:pd.isnull(row).sum()==0 and row.Sale_Price>100,1)]

new_dataset=pd.get_dummies(filtered_df[dummy_columns])
final_dataset=pd.concat([filtered_df[y_columns+num_columns+join_columns],new_dataset],1)
subsampled_df=final_dataset.sample(frac =0.2)
final_dataset_X=subsampled_df.drop(columns=y_columns)
final_dataset_Y=subsampled_df[y_columns[0]]

#Multiple Regression Model, Author: mrezhepova3
x_train, x_test, y_train, y_test = train_test_split(final_dataset_X, final_dataset_Y, test_size=0.20, random_state=614)
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

pred_test = regr.predict(x_test)
pred_train = regr.predict(x_train)

abs_err = abs(pred_test - y_test)

pred_scaled=pred_test/y_test 
y_test_scaled=y_test/y_test

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, pred_test))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))
print("R2 score:",metrics.r2_score(y_test, pred_test))



