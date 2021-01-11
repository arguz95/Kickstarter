#Individual Project - INSY662

#Author: Arnaud Guzman-Annès
#Date: November 18, 2020

############### 1. Regression Model ###############

'''
1. Develop a regression model (i.e., a supervised-learning model where the 
target variable is a continuous variable) to predict the value of the variable 
“usd_pledged.” After you obtain the final model, explain the model and justify 
the predictors you include/exclude.
'''

## Developing the model ###

# Load pandas and numpy
import pandas
import numpy

# Import Data
kickstarter_df = pandas.read_excel("Kickstarter.xlsx")

# Clean and pre-select Data

# Create goal in USD by converting goal with USD rate and drop 'goal' and 'static_usd_rate'. Drop them after.
kickstarter_df['goal_usd'] = kickstarter_df['goal'] * kickstarter_df['static_usd_rate']
kickstarter_df = kickstarter_df.drop(columns=['goal','static_usd_rate'])

# Drop labels
kickstarter_df = kickstarter_df.drop(columns=['project_id','name'])
# Drop 'launch_to_state_change_days' because it has null values
kickstarter_df = kickstarter_df.drop(columns=['launch_to_state_change_days'])
# Keep successful and failed projects
kickstarter_df = kickstarter_df.loc[(kickstarter_df['state'] == 'successful') | (kickstarter_df['state'] == 'failed') ]
# Drop predictors obtained after project is launched
kickstarter_df = kickstarter_df.drop(columns=['staff_pick','state','backers_count','pledged','spotlight'])
# Drop predictors with date format
kickstarter_df = kickstarter_df.drop(columns=['deadline','state_changed_at','created_at','launched_at'])
# Drop unnecessary predictors - Part 1
kickstarter_df = kickstarter_df.drop(columns=['currency','disable_communication','name_len','blurb_len'])
# Drop unnecessary predictors - Part 2
kickstarter_df = kickstarter_df.drop(columns=['state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr'])
# Drop missing observations
kickstarter_df = kickstarter_df.dropna()

# Dummify

# Dummify variables - Part 1
kickstarter_df = pandas.get_dummies(kickstarter_df)
# Dummify variables - Part 2
kickstarter_df = pandas.get_dummies(kickstarter_df, columns = ['deadline_month','deadline_day','deadline_yr','deadline_hr'])
# Dummify variables - Part 3
kickstarter_df = pandas.get_dummies(kickstarter_df, columns = ['created_at_month','created_at_day','created_at_yr','created_at_hr'])
# Dummify variables - Part 4
kickstarter_df = pandas.get_dummies(kickstarter_df, columns = ['launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])

##############################################################################
# Feature Selection
X = kickstarter_df.loc[:, kickstarter_df.columns != 'usd_pledged']
y = kickstarter_df['usd_pledged']

##############################################################################
# Removing anomalies with Isolation Forest

# Create isolation forest model
# Contamination parameter of 0.05 to get ~700 (~5%) of total anomalies (14214)
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100, contamination=.05, random_state = 5)
pred = iforest.fit_predict(kickstarter_df)
score = iforest.decision_function(kickstarter_df)

# Extracting anomalies
from numpy import where 
anom_index = where(pred==-1)
values = kickstarter_df.iloc[anom_index]

# Remove anomalies from dataset
kickstarter_df = pandas.concat([kickstarter_df, values, values]).drop_duplicates(keep=False)
X = kickstarter_df.loc[:, kickstarter_df.columns != 'usd_pledged']
y = kickstarter_df['usd_pledged']

##############################################################################
# Random forest for feature selection

from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state = 5)
model_regression = randomforest.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model_regression)
sfm.fit(X, y)   
coef_rf = pandas.DataFrame(list(zip(X.columns,model_regression.feature_importances_)), columns = ['Predictor','Gini index']).sort_values('Gini index',ascending = False)
print(coef_rf)

##############################################################################
# Select relevant variables - We pick those with Gini coeff > 0.01
X = X[['goal_usd','deadline_hr_14','create_to_launch_days','name_len_clean','blurb_len_clean','launch_to_deadline_days','launched_at_hr_8','deadline_day_8','launched_at_day_21','launched_at_day_26','created_at_day_16','created_at_day_7','category_Wearables']]

##############################################################################
#K-fold
##############################################################################
# CART - cross validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score 
model1 = DecisionTreeRegressor(random_state = 5, max_depth = 4)
score1 = cross_val_score(estimator=model1, X=X, y=y, scoring = 'neg_mean_squared_error', cv=5) 
mse1 = numpy.average(score1)
print(mse1)

##############################################################################
# Random Forest - cross validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score 
model2 = RandomForestRegressor(random_state = 5, max_features = 4) 
score2 = cross_val_score(estimator=model2, X=X, y=y, scoring = 'neg_mean_squared_error', cv=5) 
mse2 = numpy.average(score2)
print(mse2)

##############################################################################
# GBM - cross validation 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score 
model3 = GradientBoostingRegressor(random_state = 5, n_estimators = 50, min_samples_split = 2)
score3 = cross_val_score(estimator=model3, X=X, y=y, scoring = 'neg_mean_squared_error', cv=5) 
mse3 = numpy.average(score3)
print(mse3)

##############################################################################
#Test - train 
##############################################################################
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

##############################################################################
#CART

# Run Decision Tree
from sklearn.tree import DecisionTreeRegressor
cart = DecisionTreeRegressor(random_state = 5, max_depth = 10) 
model4 = cart.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model4.predict(X_test)
from sklearn.metrics import mean_squared_error
mse4 = mean_squared_error(y_test, y_test_pred)
print(mse4)

##############################################################################
#Random Forest

# Run Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 5, max_depth = 4) 
model5 = rf.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model5.predict(X_test)
# Calculate the mean squared error of the prediction 
mse5 = mean_squared_error(y_test, y_test_pred) 
print(mse5)

##############################################################################
#GBM

# Run Random Forest
from sklearn.ensemble import GradientBoostingRegressor
gbt = GradientBoostingRegressor(random_state = 5, n_estimators = 100) 
model6 = gbt.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model6.predict(X_test)
# Calculate the mean squared error of the prediction 
mse6 = mean_squared_error(y_test, y_test_pred) 
print(mse6)

##############################################################################

#Summary
summary = pandas.DataFrame(list(zip(['CART-cross_val','Random Forest-cross_val','GBT-cross_val',
                                     'CART-train_test','Random Forest-train_test','GBT-train_test'],
                                    [mse1*-1,mse2*-1,mse3*-1,mse4,mse5,mse6])), columns = ['Model','MSE'])
summary

##############################################################################
## Grading ##

kickstarter_grading_df = pandas.read_excel("Kickstarter-Grading.xlsx")

# Clean and pre-select Data

# Create goal in USD by converting goal with USD rate and drop 'goal' and 'static_usd_rate'. Drop them after.
kickstarter_grading_df['goal_usd'] = kickstarter_grading_df['goal'] * kickstarter_grading_df['static_usd_rate']
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['goal','static_usd_rate'])

# Drop labels
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['project_id','name'])
# Drop 'launch_to_state_change_days' because it has null values
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['launch_to_state_change_days'])
# Keep successful and failed projects
kickstarter_grading_df = kickstarter_grading_df.loc[(kickstarter_grading_df['state'] == 'successful') | (kickstarter_grading_df['state'] == 'failed') ]
# Drop predictors obtained after project is launched
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['staff_pick','state','backers_count','pledged','spotlight'])
# Drop predictors with date format
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['deadline','state_changed_at','created_at','launched_at'])
# Drop unnecessary predictors - Part 1
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['currency','disable_communication','name_len','blurb_len'])
# Drop unnecessary predictors - Part 2
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr'])
# Drop missing observations
kickstarter_grading_df = kickstarter_grading_df.dropna()

# Dummify

# Dummify variables - Part 1
kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df)
# Dummify variables - Part 2
kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df, columns = ['deadline_month','deadline_day','deadline_yr','deadline_hr'])
# Dummify variables - Part 3
kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df, columns = ['created_at_month','created_at_day','created_at_yr','created_at_hr'])
# Dummify variables - Part 4
kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df, columns = ['launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])

X = kickstarter_grading_df[['goal_usd','deadline_hr_14','create_to_launch_days','name_len_clean','blurb_len_clean','launch_to_deadline_days','launched_at_hr_8','deadline_day_8','launched_at_day_21','launched_at_day_26','created_at_day_16','created_at_day_7','category_Wearables']]
y = kickstarter_grading_df['usd_pledged']

##############################################################################
X_grading = X
y_grading = y

# GBT - cross validation 
model_grading = GradientBoostingRegressor(random_state = 5, n_estimators = 50, min_samples_split = 2)
score_grading = cross_val_score(estimator=model_grading, X=X_grading, y=y_grading, scoring = 'neg_mean_squared_error', cv=5) 

##############################################################################
# Calculate the MSE
mse_grading = numpy.average(score_grading)
mse_grading

#END - Part 1
##############################################################################
#%%

############### 2. Classification Model ###############

'''
2. Develop a classification model (i.e., a supervised-learning model where the 
target variable is a categorical variable) to predict whether the variable 
“state” will take the value “successful” or “failure.” After you obtain the 
final model, explain the model and justify the predictors you include/exclude.
'''

## Developing the model ###

# Load pandas
import pandas

# Import Data
kickstarter_df = pandas.read_excel("Kickstarter.xlsx")

# Clean and pre-select Data

# Create goal in USD by converting goal with USD rate and drop 'goal' and 'static_usd_rate'. Drop them after.
kickstarter_df['goal_usd'] = kickstarter_df['goal'] * kickstarter_df['static_usd_rate']
kickstarter_df = kickstarter_df.drop(columns=['goal','static_usd_rate'])

# Drop labels
kickstarter_df = kickstarter_df.drop(columns=['project_id','name'])
# Drop 'launch_to_state_change_days' because it has null values
kickstarter_df = kickstarter_df.drop(columns=['launch_to_state_change_days'])
# Keep successful and failed projects
kickstarter_df = kickstarter_df.loc[(kickstarter_df['state'] == 'successful') | (kickstarter_df['state'] == 'failed') ]
# Drop predictors obtained after project is launched
kickstarter_df = kickstarter_df.drop(columns=['staff_pick','usd_pledged','backers_count','pledged','spotlight'])
# Drop predictors with date format
kickstarter_df = kickstarter_df.drop(columns=['deadline','state_changed_at','created_at','launched_at'])
# Drop unnecessary predictors - Part 1
kickstarter_df = kickstarter_df.drop(columns=['currency','disable_communication','name_len','blurb_len'])
# Drop unnecessary predictors - Part 2
kickstarter_df = kickstarter_df.drop(columns=['state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr'])
# Drop missing observations
kickstarter_df = kickstarter_df.dropna()

# Dummify

# Convert to binary: successfull=1 and failed=0 for state
kickstarter_df['state'] = kickstarter_df['state'].map({'successful': 1, 'failed': 0})

# Dummify variables - Part 1
kickstarter_df = pandas.get_dummies(kickstarter_df)
# Dummify variables - Part 2
kickstarter_df = pandas.get_dummies(kickstarter_df, columns = ['deadline_month','deadline_day','deadline_yr','deadline_hr'])
# Dummify variables - Part 3
kickstarter_df = pandas.get_dummies(kickstarter_df, columns = ['created_at_month','created_at_day','created_at_yr','created_at_hr'])
# Dummify variables - Part 4
kickstarter_df = pandas.get_dummies(kickstarter_df, columns = ['launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])

##############################################################################
# Feature Selection
X = kickstarter_df.loc[:, kickstarter_df.columns != 'state']
y = kickstarter_df['state']

##############################################################################
# Removing anomalies with Isolation Forest

# Create isolation forest model
# Contamination parameter of 0.05 to get ~700 (~5%) of total anomalies (14214)
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100, contamination=.05, random_state = 5)
pred = iforest.fit_predict(kickstarter_df)
score = iforest.decision_function(kickstarter_df)

# Extracting anomalies
from numpy import where 
anom_index = where(pred==-1)
values = kickstarter_df.iloc[anom_index]

# Remove anomalies from dataset
kickstarter_df = pandas.concat([kickstarter_df, values, values]).drop_duplicates(keep=False)
X = kickstarter_df.loc[:, kickstarter_df.columns != 'state']
y = kickstarter_df['state']

##############################################################################
# Random forest for feature selection

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state = 5)
model_classifier = randomforest.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model_classifier)
sfm.fit(X, y)   
coef_rf = pandas.DataFrame(list(zip(X.columns,model_classifier.feature_importances_)), columns = ['predictor','Gini_coefficient']).sort_values('Gini_coefficient',ascending = False)
print(coef_rf)

##############################################################################
# Select relevant variables - We pick those with Gini coeff > 0.005
X = X[['goal_usd','create_to_launch_days','name_len_clean','category_Web','launch_to_deadline_days','blurb_len_clean','category_Software','category_Plays','category_Musical','category_Festivals','launched_at_weekday_Tuesday','country_US','created_at_weekday_Monday','category_Hardware','created_at_weekday_Tuesday','created_at_weekday_Wednesday']]

##############################################################################
# Split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

##############################################################################
#CART

# Run Decision Tree
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier(random_state=5, max_depth = 10) 
model1 = cart.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model1.predict(X_test)

from sklearn import metrics
accuracy1 = metrics.accuracy_score(y_test, y_test_pred)
precision1 = metrics.precision_score(y_test, y_test_pred)
recall1 = metrics.recall_score(y_test, y_test_pred)
f11 = metrics.f1_score(y_test, y_test_pred)

##############################################################################
#Random Forest

# Run Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=5, max_depth=4) 
model2 = rf.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model2.predict(X_test)

from sklearn import metrics
accuracy2 = metrics.accuracy_score(y_test, y_test_pred)
precision2 = metrics.precision_score(y_test, y_test_pred)
recall2 = metrics.recall_score(y_test, y_test_pred)
f12 = metrics.f1_score(y_test, y_test_pred)

##############################################################################
#GBM

from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier(random_state=5, n_estimators = 100)

model3 = gbt.fit(X_train, y_train) 
y_test_pred = model3.predict(X_test)

from sklearn import metrics
accuracy3 = metrics.accuracy_score(y_test, y_test_pred)
precision3 = metrics.precision_score(y_test, y_test_pred)
recall3 = metrics.recall_score(y_test, y_test_pred)
f13 = metrics.f1_score(y_test, y_test_pred)

##############################################################################

summary1 = pandas.DataFrame(list(zip(['Accuracy', 'Precision', 'Recall', 'F1'],
                                    [accuracy1, precision1, recall1, f11])), columns = ['Metric','Score'])
print(summary1)

summary2 = pandas.DataFrame(list(zip(['Accuracy', 'Precision', 'Recall', 'F1'],
                                    [accuracy2, precision2, recall2, f12])), columns = ['Metric','Score'])
print(summary2)

summary3 = pandas.DataFrame(list(zip(['Accuracy', 'Precision', 'Recall', 'F1'],
                                    [accuracy3, precision3, recall3, f13])), columns = ['Metric','Score'])
print(summary3)

##############################################################################
## Grading ##

kickstarter_grading_df = pandas.read_excel("Kickstarter-Grading.xlsx")

# Clean and pre-select Data

# Create goal in USD by converting goal with USD rate and drop 'goal' and 'static_usd_rate'. Drop them after.
kickstarter_grading_df['goal_usd'] = kickstarter_grading_df['goal'] * kickstarter_grading_df['static_usd_rate']
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['goal','static_usd_rate'])

# Drop labels
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['project_id','name'])
# Drop 'launch_to_state_change_days' because it has null values
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['launch_to_state_change_days'])
# Keep successful and failed projects
kickstarter_grading_df = kickstarter_grading_df.loc[(kickstarter_grading_df['state'] == 'successful') | (kickstarter_grading_df['state'] == 'failed') ]
# Drop predictors obtained after project is launched
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['staff_pick','usd_pledged','backers_count','pledged','spotlight'])
# Drop predictors with date format
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['deadline','state_changed_at','created_at','launched_at'])
# Drop unnecessary predictors - Part 1
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['currency','disable_communication','name_len','blurb_len'])
# Drop unnecessary predictors - Part 2
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr'])
# Drop missing observations
kickstarter_grading_df = kickstarter_grading_df.dropna()

# Dummify

# Convert to binary: successfull=1 and failed=0 for state
kickstarter_grading_df['state'] = kickstarter_grading_df['state'].map({'successful': 1, 'failed': 0})

# Dummify variables - Part 1
kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df)
# Dummify variables - Part 2
kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df, columns = ['deadline_month','deadline_day','deadline_yr','deadline_hr'])
# Dummify variables - Part 3
kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df, columns = ['created_at_month','created_at_day','created_at_yr','created_at_hr'])
# Dummify variables - Part 4
kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df, columns = ['launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])

X = kickstarter_grading_df[['goal_usd','create_to_launch_days','name_len_clean','category_Web','launch_to_deadline_days','blurb_len_clean','category_Software','category_Plays','category_Musical','category_Festivals','launched_at_weekday_Tuesday','country_US','created_at_weekday_Monday','category_Hardware','created_at_weekday_Tuesday','created_at_weekday_Wednesday']]
y = kickstarter_grading_df['state']

##############################################################################
X_grading = X
y_grading = y

model_selected = model3

# GBT - Test-train
y_grading_pred = model_selected.predict(X_grading)
##############################################################################
# Calculate the MSE
metrics.accuracy_score(y_test, y_test_pred)

#END - Part 2
##############################################################################
#%%

############### 3. Clustering Model ###############

'''
3. Develop a clustering model (i.e., an unsupervised-learning model which can 
group observations together) to group projects together. After you obtain the 
final clusters, explain the characteristics that you observe in each cluster.
'''
## Developing the model ###

# Load pandas and numpy
import pandas
import numpy

# Import Data
kickstarter_df = pandas.read_excel("Kickstarter.xlsx")

# Clean and pre-select Data

# Create goal in USD by converting goal with USD rate and drop 'goal' and 'static_usd_rate'. Drop them after.
kickstarter_df['goal_usd'] = kickstarter_df['goal'] * kickstarter_df['static_usd_rate']
kickstarter_df = kickstarter_df.drop(columns=['goal','static_usd_rate'])

# Drop labels
kickstarter_df = kickstarter_df.drop(columns=['project_id','name'])
# Drop 'launch_to_state_change_days' because it has null values
kickstarter_df = kickstarter_df.drop(columns=['launch_to_state_change_days'])
# Keep successful and failed projects
kickstarter_df = kickstarter_df.loc[(kickstarter_df['state'] == 'successful') | (kickstarter_df['state'] == 'failed') ]
# Drop predictors with date format
kickstarter_df = kickstarter_df.drop(columns=['deadline','state_changed_at','created_at','launched_at'])
# Drop unnecessary predictors - Part 1
kickstarter_df = kickstarter_df.drop(columns=['currency','disable_communication','name_len','blurb_len'])
# Drop unnecessary predictors - Part 2
kickstarter_df = kickstarter_df.drop(columns=['state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr'])
# Drop missing observations
kickstarter_df = kickstarter_df.dropna()

# Dummify

# Convert to binary: successfull=1 and failed=0 for state
kickstarter_df['state'] = kickstarter_df['state'].map({'successful': 1, 'failed': 0})

# Dummify variables - Part 1
kickstarter_df = pandas.get_dummies(kickstarter_df)
# Dummify variables - Part 2
kickstarter_df = pandas.get_dummies(kickstarter_df, columns = ['spotlight','staff_pick'])

##############################################################################
# Removing anomalies with Isolation Forest

# Create isolation forest model
# Contamination parameter of 0.05 to get ~700 (~5%) of total anomalies (14214)
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100, contamination=.05, random_state = 5)
pred = iforest.fit_predict(kickstarter_df)
score = iforest.decision_function(kickstarter_df)

# Extracting anomalies
from numpy import where 
anom_index = where(pred==-1)
values = kickstarter_df.iloc[anom_index]

#Remove anomalies from dataset
kickstarter_df = pandas.concat([kickstarter_df, values, values]).drop_duplicates(keep=False)

##############################################################################
# Feature Selection
X = kickstarter_df.loc[:, kickstarter_df.columns != 'usd_pledged']
y = kickstarter_df['usd_pledged']

##############################################################################
# Random forest for feature selection

from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state = 5)
model_classifier = randomforest.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model_classifier)
sfm.fit(X, y)   
coef_rf = pandas.DataFrame(list(zip(X.columns,model_classifier.feature_importances_)), columns = ['predictor','Gini_coefficient']).sort_values('Gini_coefficient',ascending = False)
print(coef_rf)

##############################################################################
# Select Variables
# We keep top 5 numerical values from Random forest selection
X = kickstarter_df[['usd_pledged','backers_count','goal_usd','name_len_clean','create_to_launch_days']]

##############################################################################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
X_std = scaler.fit_transform(X)

from sklearn.cluster import KMeans

#Elbow Method to determine number of optimal clusters
withinss = []
for i in range (2,10):
    kmeans = KMeans(n_clusters=i) 
    model = kmeans.fit(X_std)
    withinss.append(model.inertia_)
    
from matplotlib import pyplot as plt
plt.plot([2,3,4,5,6,7,9,10],withinss)
plt.xlabel("Clusters")
plt.ylabel("Inertia")

## We choose n_cluster = 5

#KMean
kmeans = KMeans(n_clusters = 5, random_state = 5) 
model = kmeans.fit(X_std)
labels = model.predict(X_std)

#Silhouette
from sklearn.metrics import silhouette_score
silhouette_score(X_std,labels)
from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_std,labels)

#Label and silhouette score for each obeservation
score_silhouette = pandas.DataFrame({'label':labels,'silhouette':silhouette})   

silhouette1 = numpy.average(score_silhouette[score_silhouette['label'] == 0].silhouette)
silhouette2 = numpy.average(score_silhouette[score_silhouette['label'] == 1].silhouette)
silhouette3 = numpy.average(score_silhouette[score_silhouette['label'] == 2].silhouette)
silhouette4 = numpy.average(score_silhouette[score_silhouette['label'] == 3].silhouette)
silhouette5 = numpy.average(score_silhouette[score_silhouette['label'] == 4].silhouette)

silhouette_array = numpy.array([silhouette1,silhouette2,silhouette3,silhouette4,silhouette5])

column_values = ['Average Silhouette score']
index_values = ["Cluster 1","Cluster 2","Cluster 3","Cluster 4","Cluster 5"]
silhouette = pandas.DataFrame(data = silhouette_array, index = index_values, columns = column_values).round(decimals = 3)
print(silhouette)

#Centroid of each cluster
cent_clust1 = kmeans.cluster_centers_[0]
cent_clust2 = kmeans.cluster_centers_[1]
cent_clust3 = kmeans.cluster_centers_[2]
cent_clust4 = kmeans.cluster_centers_[3]
cent_clust5 = kmeans.cluster_centers_[4]

cluster_array = numpy.array([cent_clust1,cent_clust2,cent_clust3,cent_clust4,cent_clust5])

column_values = [X]
index_values = ["Cluster 1","Cluster 2","Cluster 3","Cluster 4","Cluster 5"]
clusters = pandas.DataFrame(data = cluster_array, index = index_values, columns = column_values).round(decimals = 2)
print(clusters)

#Count
kickstarter_df['cluster'] = model.labels_
kickstarter_df['cluster'].value_counts()

'''
Cluster 1: 467
Cluster 2: 5709
Cluster 3: 6922
Cluster 4: 3
Cluster 5: 402
'''

##############################################################################

#X = kickstarter_df[['usd_pledged','backers_count','goal_usd','name_len_clean','create_to_launch_days']]

# Relevant Plots

###
plt.scatter(X['backers_count'],X['usd_pledged'],c = labels, cmap = 'rainbow')
plt.xlabel("backers")
plt.ylabel("pledged")

plt.scatter(X['backers_count'],X['goal_usd'],c = labels, cmap = 'rainbow')
plt.xlabel("backers")
plt.ylabel("goal")

plt.scatter(X['backers_count'],X['name_len_clean'],c = labels, cmap = 'rainbow')
plt.xlabel("backers")
plt.ylabel("name lenght")

plt.scatter(X['backers_count'],X['create_to_launch_days'],c = labels, cmap = 'rainbow')
plt.xlabel("backers")
plt.ylabel("launch days")

###
plt.scatter(X['usd_pledged'],X['goal_usd'],c = labels, cmap = 'rainbow')
plt.xlabel("pledged")
plt.ylabel("goal")

plt.scatter(X['usd_pledged'],X['name_len_clean'],c = labels, cmap = 'rainbow')
plt.xlabel("pledged")
plt.ylabel("name")

plt.scatter(X['usd_pledged'],X['create_to_launch_days'],c = labels, cmap = 'rainbow')
plt.xlabel("pledged")
plt.ylabel("launch days")

###
plt.scatter(X['goal_usd'],X['name_len_clean'],c = labels, cmap = 'rainbow')
plt.xlabel("goal")
plt.ylabel("name")

plt.scatter(X['goal_usd'],X['create_to_launch_days'],c = labels, cmap = 'rainbow')
plt.xlabel("spot")
plt.ylabel("launch days")
###

plt.scatter(X['name_len_clean'],X['create_to_launch_days'],c = labels, cmap = 'rainbow')
plt.xlabel("name")
plt.ylabel("launch")

#END - Part 3 - #END Individual Project
##############################################################################