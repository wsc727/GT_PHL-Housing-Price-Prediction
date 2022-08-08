# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:27:38 2020

@author: wsc72
"""
#pip install pyarrow 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import time
import seaborn as sns
from sklearn import metrics
import datetime 

df_CPI = pd.read_excel('BLS CPI U PHL 1980-2020.xlsx', header = 11, usecols = "A:M")
df_CPI.index = df_CPI["Year"]
df_CPI_months=df_CPI.drop(columns=["Year"])
df_CPI_months=df_CPI_months.stack().reset_index()
df_CPI_months.columns=["Year","Month","CPI"]
df_CPI_months["Year"]=df_CPI_months.Year.astype(str)

df_unemployment = pd.read_excel('BLS PHL Unemployment 1990-2020.xlsx', header = 10, usecols = "A:M")
df_unemployment.index = df_unemployment["Year"]
df_unemployment_months=df_unemployment.drop(columns=["Year"])
df_unemployment_months=df_unemployment_months.stack().reset_index()
df_unemployment_months.columns=["Year","Month","unemployment"]
df_unemployment_months["Year"]=df_unemployment_months.Year.astype(str)   

df_interest_rate=pd.read_excel("Federal Reserve Interest Rates Data.xlsx",header=0,usecols="A:L")
df_interest_rate["Year"]=df_interest_rate.Month.apply(lambda x:x.split("-")[0])
df_interest_rate["Month"]=df_interest_rate.Month.apply(lambda x:datetime.datetime.strptime(x.split("-")[-1], "%m").strftime("%b"))
df_interest_rate=df_interest_rate[['Year','Month','Federal Funds Rate','PRIME Rate']]
df_interest_rate.columns=['Year','Month','Federal-Funds-Rate','PRIME-Rate']

#merge
df_merge=pd.merge(df_CPI_months,df_unemployment_months,left_on=["Year","Month"],right_on=["Year","Month"],how="outer")
df_merge_final=pd.merge(df_merge,df_interest_rate,left_on=["Year","Month"],right_on=["Year","Month"],how="outer")
df_merge_final=df_merge_final.sort_values(by=["Year","Month"])

#major data
data_df_main=pd.read_excel('PHL_assessment_clean.xlsx', sheet_name="PHL_assessment_clean",header = 0, usecols = "A:Y")
data_df_main["Year"]=data_df_main.Sale_Date.apply(lambda x:x.year)
data_df_main["Sale_Month"]=data_df_main.Sale_Date.apply(lambda x:datetime.datetime.strptime(str(x.month), "%m").strftime("%b"))
data_df_main["Sale_Year"]=data_df_main.Year.astype(str)

data_df_main.Zip_Code = data_df_main.Zip_Code.apply(lambda x:str(x)[:5])
#final df
final_df=pd.merge(data_df_main,df_merge_final,left_on=["Sale_Year","Sale_Month"],right_on=["Year","Month"],how="outer")

def cal_age(row):
    if pd.isnull(row["Sale_Year"]) or pd.isnull(row["Year_Built"]) or row["Year_Built"] == '196Y':
        age=np.nan
    else:
        age=int(row.Sale_Year)-int(row.Year_Built)
    return(age)


final_df["age"]=final_df.apply(lambda x:cal_age(x),1)
final_df = final_df[final_df['Bedroom_Count'] <7] # removed ~2100 rows
# try single family only

final_df = final_df[final_df['Building_Category'] == 'Single Family']
#247775 rows left

# Remove unit, 'Construction_Quality', 
y_columns=["Sale_Price"]
dummy_columns=[ 'Physical_Condition', 'Air_Conditioning', 'Aux_Unit', 'Elevation_Type', 'Zip_Code']

num_columns=["Sale_Year",'Bedroom_Count', 'Full_Bathroom_Count', 'Total_Room_Count', 'Lot_Size', 'Number_of_Stories', 'Fireplaces_Count', 'Basement_Garage_Parking_Space_Count', 'Basement_Room_Count', 'Non_Basement_Area']
join_columns=['CPI', 'unemployment', 'Federal-Funds-Rate','PRIME-Rate', 'age']
filtered_df=final_df[y_columns+dummy_columns+num_columns+join_columns] #keep original one
#remove row with NA #remove row with age less than 0 and 
filtered_df=filtered_df[filtered_df.apply(lambda row:pd.isnull(row).sum()==0 and row.age>=0 and row.Sale_Price>100,1)]

zipcode_morethan_500=list(filtered_df["Zip_Code"].value_counts()[filtered_df["Zip_Code"].value_counts()>500].index)
filtered_df_final=filtered_df[filtered_df.apply(lambda x:x.Zip_Code in zipcode_morethan_500,1)]   
print(filtered_df_final.shape)
#247772

new_dataset=pd.get_dummies(filtered_df[dummy_columns])
final_dataset=pd.concat([filtered_df[y_columns+num_columns+join_columns],new_dataset],1)

final_dataset_X=final_dataset.drop(columns=y_columns)
final_dataset_Y=final_dataset[y_columns[0]]

## change hyperparameter combo

#Use this
#remove price >10 million and single family only 
final_dataset = final_dataset[final_dataset.Sale_Price < 10000000]
#247114 rows
final_dataset_X=final_dataset.drop(columns=y_columns)
final_dataset_Y=final_dataset[y_columns[0]]
X_train, X_test, y_train, y_test = train_test_split(final_dataset_X, final_dataset_Y, test_size=0.3)


rf_final_sqt = RandomForestRegressor(oob_score=True, max_depth= 100, max_features= 'sqrt', n_estimators= 500, verbose=3) #turn on oob_score, for  regressor, defrault is R^2 score
final_fit_less = rf_final_sqt.fit(X_train, y_train)
#5.1min

pred_less = final_fit_less.predict(X_test)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred_less)))
#~159480.1
print("R2 score:",metrics.r2_score(y_test, pred_less))
#~74.3%
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, pred_less))
#54500.2
pred_less_train = final_fit_less.predict(X_train)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_train, pred_less_train)))
#66558.9
print("R2 score:",metrics.r2_score(y_train, pred_less_train))
#~0.957
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_train, pred_less_train))
#20904

PHL_name = X_train.columns
PHL =  final_fit_less.feature_importances_
df = {PHL_name[i]: PHL[i] for i in range(len(PHL_name))}
df_PHL = pd.DataFrame(df, index=[0])
df_PHL.to_csv('PHL.csv')

#assume saled in 2020-12-31
df_2020 = data_df_main
df_2020['Sale_Date']='2020-12-31'
df_2020['Sale_Date'] = pd.to_datetime(df_2020['Sale_Date'])

df_2020["Year"]=df_2020.Sale_Date.apply(lambda x:x.year)
df_2020["Sale_Month"]=df_2020.Sale_Date.apply(lambda x:datetime.datetime.strptime(str(x.month), "%m").strftime("%b"))
df_2020["Sale_Year"]=df_2020.Year.astype(str)
final_2020=pd.merge(df_2020, df_merge_final,left_on=["Sale_Year","Sale_Month"],right_on=["Year","Month"],how="outer")

final_2020["age"]=final_2020.apply(lambda x:cal_age(x),1)
final_2020 = final_2020[final_2020['Bedroom_Count'] <7] 
final_2020["PRIME-Rate"]=final_2020["PRIME-Rate"].fillna(3.25)
final_2020["Federal-Funds-Rate"]=final_2020["Federal-Funds-Rate"].fillna(0.09)
final_2020 = final_2020[final_2020['Building_Category'] == 'Single Family']
id_column=["Parcel_ID"]
filtered_2020=final_2020[y_columns+dummy_columns+num_columns+join_columns+id_column] 
filtered_2020=filtered_2020[filtered_2020.apply(lambda row:pd.isnull(row).sum()==0 and row.age>=0 and row.Sale_Price>100,1)]

zipcode_morethan_500=list(filtered_2020["Zip_Code"].value_counts()[filtered_2020["Zip_Code"].value_counts()>500].index)
filtered_df_2020=filtered_2020[filtered_2020.apply(lambda x:x.Zip_Code in zipcode_morethan_500,1)]

filtered_2020 = filtered_df_2020[filtered_df_2020.Sale_Price < 10000000]

#check cats value number and do clean
new_2020=pd.get_dummies(filtered_2020[dummy_columns])
final_2020data=pd.concat([filtered_2020[y_columns+num_columns+join_columns+id_column],new_2020],1)

X_2020 = final_2020data.drop(columns=y_columns)
#remove the rows and feautreus not used in training
X_2020['Zip_Code_19110']=0
X_2020['Zip_Code_19113']=0
X_2020 = X_2020.drop(columns=id_column)
#remove_columns=set(X_train.columns) -set(X_2020.columns)-set(id_column)
#drop_index=X_2020.apply(lambda x:sum(x[remove_columns])==0,1)

prediction_2020 = final_fit_less.predict(X_2020)
final_2020data['prediction_2020DecSale'] = prediction_2020
final_2020data.to_csv("X_2020_filteredPHL.csv")
original = data_df_main
predict = pd.read_csv('X_2020_filteredPHL.csv')
predict_f = predict[['Parcel_ID', 'prediction_2020DecSale']]
df_tableau=pd.merge(original,predict_f,left_on=["Parcel_ID"],right_on=["Parcel_ID"],how="inner")
df_tableau.to_csv('RF_PHL_for_tableau.csv')
