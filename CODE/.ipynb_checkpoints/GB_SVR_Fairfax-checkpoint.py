import pandas as pd
import numpy as np
import requests
import fastparquet as fp
from datetime import datetime
import time
from math import sin, cos, sqrt, atan2, radians
import time
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy import stats
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


assessment = fp.ParquetFile('Fairfax_County_VA_clean.parq').to_pandas()

unemployment = pd.read_excel('PHL_unemployment.xlsx')
int_rates = pd.read_excel('Federal Reserve Interest Rates Data.xlsx')
stock_price = pd.read_excel('Russell 3000 Monthly.xlsx')

#minor adjustments for merge
unemployment['Date_BLS'] = pd.to_datetime(unemployment['Date_BLS'], infer_datetime_format=True)
int_rates['Date_prime'] = pd.to_datetime(int_rates['Date_prime'], infer_datetime_format=True)
stock_price['Date_Stock'] = pd.to_datetime(stock_price['Date_Stock'], infer_datetime_format=True)

assessment['Sale_month'] = assessment['Sale_Date'].apply(lambda x: x.replace(day=1))

#merge economic data to main dataset
assessment = assessment.merge(unemployment,left_on='Sale_month',right_on='Date_BLS',how='left')
assessment = assessment.merge(int_rates,left_on='Sale_month',right_on='Date_prime',how='left')
assessment = assessment.merge(stock_price,left_on='Sale_month',right_on='Date_Stock',how='left')


assessment_train = assessment[assessment['Non_Basement_Area'].notna()]
assessment_train['year'] = assessment_train['Sale_Date'].dt.year
assessment_train['month'] = assessment_train['Sale_Date'].dt.month
assessment_train = assessment_train[assessment_train['year']>=2000]
assessment_train = assessment_train[(assessment_train['Sale_Price']<=2000000) & (assessment_train['Sale_Price']>=1000)]

assessment_train['Air_Conditioning_n'] = np.where(assessment_train['Air_Conditioning']=='Central A/C',1,0)
assessment_train['Basement_Room_Count'] = assessment_train['Basement_Room_Count'].astype(int)

assessment_train.drop(columns=['Date_BLS','Date_prime','Date_Stock'],inplace=True)
assessment_train.drop(columns=['Parcel_ID','Street_Number','Street_Number_Suffix','Street_Direction','Actual_Street_Name','Street_Type','Street_Name_Suffix','Year_Remodeled','Air_Conditioning','Sale_Date','Tax_District_Description','Sale_month','lat','lng'],inplace=True)
#change strings to ints to preprocess for ML algo
def strnums(cols):
    return dict(zip(set(assessment_train[cols]),list(range(0,len(set(assessment_train[cols]))))))
for columns in set(assessment_train.select_dtypes(exclude=['number','float64','int64'])):
    assessment_train[columns] = assessment_train[columns].map(strnums(columns))

    
assessment_train = assessment_train.dropna()
assessment_train = assessment_train.sort_values(by = 'year',ascending=True)

#visual exploration of data
sns.distplot(assessment_train['Sale_Price'])

var = 'Non_Basement_Area'
data = pd.concat([assessment_train['Sale_Price'], assessment_train[var]], axis=1)
data.plot.scatter(x=var, y='Sale_Price', ylim=(0,1500000), s=32)

var = 'Non_Basement_Area'
data = pd.concat([assessment_train['Sale_Price'], assessment_train[var]], axis=1)
data.plot.scatter(x=var, y='Sale_Price', ylim=(0,1500000), s=32)

var = 'Physical_Condition'
data = pd.concat([assessment_train['Sale_Price'], assessment_train[var]], axis=1)
f, ax = plt.subplots(figsize=(14, 8))
fig = sns.boxplot(x=var, y="Sale_Price", data=data)
fig.axis(ymin=0, ymax=1500000)

cols = ['Sale_Price', 'Non_Basement_Area', 'Full_Bathroom_Count', 'prime_rate']
sns.pairplot(assessment_train[cols], size = 4)

num_corr=assessment_train.corr()
k = 21
cols = num_corr.nlargest(k, 'Sale_Price')['Sale_Price'].index
cm = np.corrcoef(assessment_train[cols].values.T)
sns.set(font_scale=1)
f, ax = plt.subplots(figsize=(13,12))
hm=sns.heatmap(cm, annot = True,vmax =.8, yticklabels=cols.values, xticklabels = cols.values,fmt=".1g")

#split training and test
y_data = assessment_train['Sale_Price']
x_data = assessment_train.drop(columns='Sale_Price')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, train_size=0.85, shuffle= False)

#gradient boost gridsearch to ge optimal parameters - do not re-run do to runtime
#params_gradient = {'n_estimators': [100, 200, 400, 600, 800],'max_depth': [3,6,9,12,15]}
# gradientr = GridSearchCV(estimator=GradientBoostingRegressor(),
#                   param_grid=paragrid_gradient,
#                   cv=5,
#                   verbose=True, n_jobs=-1)
# gradientr.fit(x_train,y_train)
# gradientr.best_params_

gradientr_clf = GradientBoostingRegressor(n_estimators=100, max_depth=9)
gradientr_clf.fit(x_train, y_train)
gradient_pred = pd.DataFrame({'actual': y_test,
                            'predicted': gradientr_clf.predict(x_test)})
print(metrics.r2_score(gradient_pred.actual, gradient_pred.predicted))
print(metrics.mean_squared_error(gradient_pred.actual, gradient_pred.predicted,squared=False))
print(metrics.mean_absolute_error(gradient_pred.actual, gradient_pred.predicted))

#gradient boost visualize feature importance
f_importance = gradientr_clf.feature_importances_
idx_sort = np.argsort(f_importance)
position = np.arange(sorted_idx.shape[0]) + .5
figure = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(position, feature_importance[idx_sort], align='center')
plt.yticks(position, np.array(x_test.columns)[idx_sort])
plt.title('Feature Importance G Boost')

result = permutation_importance(gradientr_clf, x_test, y_test, n_repeats=10, n_jobs=2)
idx_sort = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[idx_sort].T,
            vert=False, labels=np.array(x_test.columns)[idx_sort])
plt.title("Permutation Importance G Boost")
fig.tight_layout()
plt.show()



#SVR
#reduce dataset size for run time purposes and re-do train/test split
assessment_reduced = assessment_train[assessment_train['year']>=2012]
y_data = assessment_reduced['Sale_Price']
x_data = assessment_reduced.drop(columns='Sale_Price')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, train_size=0.8, shuffle= False)

#SVR scaling and PCA
transf = StandardScaler()
scaled_x_train = pd.DataFrame(transf.fit(x_train).transform(x_train),columns = x_train.columns)
scaled_x_test = pd.DataFrame(transf.fit(x_train).transform(x_test),columns = x_test.columns)
        
n_col = scaled_x_train.shape[1]
pca = PCA(n_components=n_col)
train_comp = pca.fit_transform(scaled_x_train)
test_comp = pca.fit_transform(scaled_x_test)
explained_var = pca.explained_variance_ratio_
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
#select 15 components
pca = PCA(n_components=15)
train_comp = pca.fit_transform(scaled_x_train)
test_comp = pca.fit_transform(scaled_x_test)

#SVR gridsearch - do not re-run do to runtime
# svm_parameters = {'kernel':('linear', 'rbf','poly'), 'C':[10,100,1000]}
# svrclf = SVR(gamma='auto',cache_size = 1000)
# svr_cv = GridSearchCV(svrclf, svm_parameters, n_jobs=-1, return_train_score=True,verbose=True)
# svr_cv.fit(train_comp,y_train2)



svrclf = SVR(kernel = 'linear',cache_size = 1000,C=1000)
svrclf.fit(train_comp, y_train)

svr_pred = pd.DataFrame({'actual': y_test,
                            'predicted': svrclf.predict(test_comp)})
print(metrics.r2_score(svr_pred.actual, svr_pred.predicted))
print(metrics.mean_squared_error(svr_pred.actual, svr_pred.predicted,squared=False))
print(metrics.mean_absolute_error(svr_pred.actual, svr_pred.predicted))
