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
from functools import partial
import time
import os,sys,pickle
import seaborn as sns
from sklearn import metrics
import datetime 
from sklearn.model_selection import GridSearchCV


os.chdir('C:\\Users\\wsc72\\Desktop\\sale data')
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
df_interest_rate["Month"]=df_interest_rate.Month.apply(lambda x:datetime.datetime.strptime(x.split("-")[-1], "%m").strftime("%b"))
df_interest_rate=df_interest_rate[['year','Month','Federal Funds Rate','PRIME Rate']]
df_interest_rate.columns=['Year','Month','Federal-Funds-Rate','PRIME-Rate']

#merge
df_merge=pd.merge(df_CPI_months,df_unemployment_months,left_on=["Year","Month"],right_on=["Year","Month"],how="outer")
df_merge_final=pd.merge(df_merge,df_interest_rate,left_on=["Year","Month"],right_on=["Year","Month"],how="outer")
df_merge_final=df_merge_final.sort_values(by=["Year","Month"])

#major data
data_df_main=pd.read_excel('Fairfax_County_VA_clean.xlsx', sheet_name="Fairfax_County_VA_clean",header = 0, usecols = "A:AA")
data_df_main["Year"]=data_df_main.Sale_Date.apply(lambda x:x.year)
data_df_main["Sale_Month"]=data_df_main.Sale_Date.apply(lambda x:datetime.datetime.strptime(str(x.month), "%m").strftime("%b"))
data_df_main["Sale_Year"]=data_df_main.Year.astype(str)

#final df
final_df=pd.merge(data_df_main,df_merge_final,left_on=["Sale_Year","Sale_Month"],right_on=["Year","Month"],how="outer")

def cal_age(row):
    if pd.isnull(row["Sale_Year"]) or pd.isnull(row["Year_Built"]):
        age=np.nan
    else:
        if pd.isnull(row.Year_Remodeled):
            age=int(row.Sale_Year)-int(row.Year_Built)
        else:
            if int(row.Year_Remodeled) > int(row.Sale_Year): #if remodel after sale, ignore
                age=int(row.Sale_Year)-int(row.Year_Built)
            else:
                later_year=max(int(row.Year_Remodeled),int(row.Sale_Year)) #select most recent between remodel and built
                age=int(row.Sale_Year)-later_year
    return(age)


final_df["age"]=final_df.apply(lambda x:cal_age(x),1)

y_columns=["Sale_Price"]
dummy_columns=['Tax_District', 'Building_Style', 'Construction_Quality', 'Physical_Condition', 'Air_Conditioning']

num_columns=["Sale_Year",'Bedroom_Count', 'Full_Bathroom_Count', 'Half_Bathroom_count', 'Basement_Size', 'Fireplaces_Count', 'Basement_Garage_Parking_Space_Count', 'Basement_Room_Count', 'Non_Basement_Area']
join_columns=['CPI', 'unemployment', 'Federal-Funds-Rate','PRIME-Rate', 'age']
filtered_df=final_df[y_columns+dummy_columns+num_columns+join_columns] #keep original one
#remove row with NA #remove row with age less than 0 and 
filtered_df=filtered_df[filtered_df.apply(lambda row:pd.isnull(row).sum()==0 and row.age>=0 and row.Sale_Price>100 and row.Bedroom_Count<8,1)]


#additonal clean, for the problematic features
#Tax_District has so many categories with low number 
#only keep more than 500
tax_districts_morethan_500=list(filtered_df["Tax_District"].value_counts()[filtered_df["Tax_District"].value_counts()>500].index)
filtered_df_final=filtered_df[filtered_df.apply(lambda x:x.Tax_District in tax_districts_morethan_500,1)]   

new_dataset=pd.get_dummies(filtered_df_final[dummy_columns])
final_dataset=pd.concat([filtered_df_final[y_columns+num_columns+join_columns],new_dataset],1)


final_dataset_X=final_dataset.drop(columns=y_columns)
final_dataset_Y=final_dataset[y_columns[0]]


###evaluate the model 
# Use the forest's predict method on the test data

rf_final = RandomForestRegressor(oob_score=True, max_depth= 100, max_features= 'sqrt', n_estimators= 900, verbose=3) #turn on oob_score, for  regressor, defrault is R^2 score

#njob = 3 memory error, 2 same
#reduce n to 900 (fail at 957)
X_train, X_test, y_train, y_test = train_test_split(final_dataset_X, final_dataset_Y, test_size=0.3)
final_fit = rf_final.fit(X_train, y_train)
#18.6 min finished 
pickle.dump(final_fit,open("final_fit_shangci.p","wb"))
predictions = final_fit.predict(X_test)
predictions_train = final_fit.predict(X_train)
# Calculate the absolute errors
errors = abs(predictions - y_test)

preditions_scaled=predictions/y_test 
y_test_scaled=y_test/y_test

len(final_fit.feature_importances_)
len(X_train.columns)
#VA_name -> col, VA -> importance
#  df = {VA_name[i]: VA[i] for i in range(len(VA_name))}
# df_VA = pd.DataFrame(df, index=[0])
# df_VA.to_csv('VA.csv')

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, predictions))
#39308.285867107785
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, predictions))

print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#189074.1288636115
print("R2 score:",metrics.r2_score(y_test, predictions))
#0.9901529416108064

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

id_column=["Parcel_ID"]
filtered_2020=final_2020[y_columns+dummy_columns+num_columns+join_columns+id_column] 
filtered_2020=filtered_2020[filtered_2020.apply(lambda row:pd.isnull(row).sum()==0 and row.age>=0 and row.Sale_Price>100,1)]
tax_districts_morethan_500=list(filtered_2020["Tax_District"].value_counts()[filtered_2020["Tax_District"].value_counts()>500].index)
filtered_df_final_2020=filtered_2020[filtered_2020.apply(lambda x:x.Tax_District in tax_districts_morethan_500,1)] 


print(filtered_df_final_2020.shape)

#270239  <- 362928 
#check cats value number and do clean


new_2020=pd.get_dummies(filtered_df_final_2020[dummy_columns])
final_2020data=pd.concat([filtered_df_final_2020[y_columns+num_columns+join_columns+id_column],new_2020],1)


X_2020 = final_2020data.drop(columns=y_columns)
#remove the rows and feautreus not used in training
remove_columns=set(X_2020.columns)-set(X_train.columns)-set(id_column)
drop_index=X_2020.apply(lambda x:sum(x[remove_columns])==0,1)
X_2020_filtered=X_2020[drop_index]
X_2020_filtered_for_training=X_2020_filtered.drop(columns=list(remove_columns)+id_column)

prediction_2020 = final_fit.predict(X_2020_filtered_for_training)
X_2020_filtered['prediction_2020DecSale'] = prediction_2020
X_2020_filtered.to_csv("X_2020_filtered.csv")








