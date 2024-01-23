# This is the main file of the project
# Team include 42441, 42450
import sqlalchemy as sql
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import numpy as np
import os

os.chdir("/Users/Ignacy/SSE files/Data Science Analytics/assignment 2/")
#Step 1

#Get the data from the SQL server
host, username, password, schema = "XXXXXXXX", "XXXX", "XXXXX" ,"XXXXXXX"
connection_string = "mysql+pymysql://{}:{}@{}/{}".format(username, password, host, schema)
connection = sql.create_engine(connection_string)

# Read the datasets

dataset = "Apartment" #Apartment:id; AnnualReport:org_numver; Housing_ 
id_name = "id"
apartment_query = "SELECT * FROM Apartment"  
report_query = "SELECT * FROM AnnualReport"
house_query = "SELECT * FROM HousingAssociation"  
apartment_df = pd.read_sql_query(con=connection.connect(), sql=sql.text(apartment_query), index_col="id")
report_df = pd.read_sql_query(con=connection.connect(), sql=sql.text(report_query), index_col="org_number")
house_df = pd.read_sql_query(con=connection.connect(), sql=sql.text(house_query), index_col="org_number")

#Save dataset files
# apartment_df.to_csv("Apartment.csv", index="id")
# report_df.to_csv("AnnualReport.csv", index="org_number")
# house_df.to_csv("HousingAsscociation.csv", index="org_number")

#Read data from the local path
# apartment_df = pd.read_csv("Apartment.csv")
# house_df = pd.read_csv("HouseAssociation.csv")
# report_df = pd.read_csv("AnnualReport.csv")

#annual report nas
#print(report_df.isna().sum())
report_df['total_loan'].replace(1, np.nan , inplace=True)
report_df['total_loan'].replace(-1, 0, inplace=True)
report_df['total_loan'].fillna(report_df['total_loan'].mean(), inplace=True)

report_df['association_tax_liability'].fillna(report_df['association_tax_liability'].mode()[0], inplace=True)
report_df['long_term_debt_other'].fillna(report_df['long_term_debt_other'].mode()[0], inplace=True)
report_df['number_of_rental_units'].fillna(report_df['number_of_rental_units'].mode()[0], inplace=True)
report_df['total_commercial_area'].fillna(report_df['total_commercial_area'].mode()[0], inplace=True)
report_df['total_plot_area'].fillna(report_df['total_plot_area'].mode()[0], inplace=True)
report_df['total_rental_area'].fillna(report_df['total_rental_area'].mode()[0], inplace=True)
report_df['savings'].fillna(report_df['savings'].mean(), inplace=True)

report_df.reset_index(inplace=True)
report_df = report_df[['org_number','association_tax_liability', 'plot_is_leased','savings', 'total_commercial_area',\
                       'total_living_area', 'total_loan','total_plot_area', 'total_rental_area']]
report_df_gb = report_df.groupby(['org_number']).agg({"association_tax_liability":lambda x: x.mode().iloc[0],
                                                      "plot_is_leased":lambda x: x.mode().iloc[0],
                                                      'savings':'mean', 'total_commercial_area':'mean',
                                                      'total_living_area':'mean', 'total_loan':'mean','total_plot_area':'mean', 'total_rental_area':'mean'})


#house association
# we should sort the construction time
# house_df = house_df[house_df['construction_year'].notna()]
# hc_stats = house_df['construction_year'].describe(percentiles=[.2, .4, .6,.8])#11, 31, 43, 85
house_df["construction_year"].fillna(-1, inplace=True)
bins = [-1, 1650, 1923, 1943, 1973, 2003, 2023]
labels = ["Unknown", "Very Old", "Old", "Medium", "New", "Very New"]
house_df['Period'] = pd.cut(house_df['construction_year'], bins=bins, labels=labels, include_lowest=True)
house_df.reset_index(inplace=True)

house_df = house_df[['org_number', 'Period']]
#print(house_df['construction_year'].value_counts())


#print(apartment_df.head())
# test = apartment_df[apartment_df['sell_price'].isna()]
#Feature Engineering
#print(apartment_df.dtypes)
#print(apartment_df.isna().sum())# get nums of NAs of each attribute

apartment_df["sell_date"] = pd.to_datetime(apartment_df["sell_date"])
apartment_df["sell_date"] = apartment_df["sell_date"].dt.year.astype(int)#only need sell year

apartment_df.rename(columns={'housing_association_org_number': 'org_number'}, inplace=True)

apm = apartment_df.copy()

#print(apm.isna().sum())
#delete useless attributes
apm.drop(["has_balcony", "width", "height", "energy_class"], axis=1, inplace=True)
#basic_elements of apartments
# apm = apartment_df[apartment_df[['rooms', 'living_area', 'rent', 'housing_association_org_number']].notna().all(axis=1)]
#standarize the categories
# apm['energy_class'].fillna(apm['energy_class'].mode()[0], inplace=True)
# apm['energy_class'].replace({'1': 'G', '2': 'F', '3': 'E', '4': 'D', '5': 'C', '6': 'B', '7': 'A'}, inplace=True)
apm['has_solar_panels'].fillna(apm['has_solar_panels'].mode()[0], inplace=True)
apm['additional_area'].fillna(0, inplace=True)
apm['plot_area'].fillna(0, inplace=True)
apm['has_fireplace'].fillna(apm['has_fireplace'].mode()[0], inplace=True)
apm['has_patio'].fillna(apm['has_patio'].mode()[0], inplace=True)
# to fill the NAs in legal district

postcode_dic = dict(zip(apm["postcode"],apm["legal_district"])) 
dic_list = sorted(postcode_dic.items()) 
dic_list = [list(pair) for pair in dic_list] 
for i in range(len(dic_list)): 
    if dic_list[i][1] == None: 
        dic_list[i][1] = dic_list[i-1][1] 
postcode_dic = dict(dic_list)     
for i in apm.index: 
    if apm.loc[i,'legal_district'] == None: 
        if apm.loc[i,'postcode'] in postcode_dic.keys(): 
            apm.loc[i,'legal_district'] = postcode_dic[apm.loc[i,'postcode']] 
        else: 
            print(apm.loc[i,'postcode'])
            
operating_per_apm = apm['operating_cost'].mean()            
operation_per_area = (apm['operating_cost']/apm['living_area']).mean()
for i in apm.index: 
    if apm.loc[i,'operating_cost'] != apm.loc[i,'operating_cost']: 
        if apm.loc[i, 'living_area'] == apm.loc[i, 'living_area']:
            apm.loc[i, 'operating_cost'] = operation_per_area*apm.loc[i, 'living_area']
        else: 
            apm.loc[i, 'operating_cost'] = operating_per_apm

apm.reset_index(inplace=True)
apm = pd.merge(apm, house_df, on='org_number', how='left')
apm = pd.merge(apm, report_df_gb, on='org_number', how='left')
apm.set_index('id',inplace=True)
# if apm['legal_district'].isna():
#     apm['legal_district'] = apm[apm['brokers_']]

apm.drop(['locality', 'brokers_description', 'postcode', 'asking_price', 'street_name', 'street_address', 'org_number', 'is_new_construction'], axis=1, inplace=True)

apm['legal_district'] = apm['legal_district'].astype('category')
apm['object_type'] = apm['object_type'].astype('category')
apm['association_tax_liability'] = apm['association_tax_liability'].astype('category')
'''
apm['has_solar_panels'] = apm['has_solar_panels'].astype('category')
apm['has_fireplace'] = apm['has_fireplace'].astype('category')
apm['has_patio'] = apm['has_patio'].astype('category')
apm['plot_is_leased'] = apm['plot_is_leased'].astype('category')

apm['has_solar_panels'] = apm['has_solar_panels'].astype('int')
apm['has_fireplace'] = apm['has_fireplace'].astype('int')
apm['has_patio'] = apm['has_patio'].astype('int')
apm['plot_is_leased'] = apm['plot_is_leased'].astype('int')
'''
val_set = apm[apm['sell_price'].isna()]


val_set['floor'].fillna(round(apm['floor'].mean()), inplace=True)
val_set['living_area'].fillna(val_set['living_area'].mean(), inplace=True)
val_set['association_tax_liability'].fillna('lowered_tax_liability', inplace = True)
val_set['plot_is_leased'].fillna(0,inplace=True)#mode
val_set['savings'].fillna(apm['savings'].mean(),inplace=True)
val_set['total_commercial_area'].fillna(apm['total_commercial_area'].mean(),inplace=True)
val_set['total_living_area'].fillna(apm['total_living_area'].mean(),inplace=True)
val_set['total_loan'].fillna(apm['total_loan'].mean(),inplace=True)
val_set['total_plot_area'].fillna(0,inplace=True)#mode
val_set['total_rental_area'].fillna(0,inplace=True)#mode

clean_set = apm.dropna()
#%%
# 
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import matplotlib.pyplot as plt
# to fill nas in room
non_room = clean_set.drop(["rent", 'sell_price','rooms'], axis=1)
room = clean_set["rooms"]
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(non_room, room, test_size=0.2, random_state=36)

# Define the XGBoost model
xgb_model = XGBRegressor()

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3]
    ,"enable_categorical":[True,True,True]
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

# Fit the model to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Make predictions on the test set using the best model
best_model_room = grid_search.best_estimator_
predictions = best_model_room.predict(X_test)
# Evaluate the model
r2 = r2_score(y_test, predictions)
print("Test R2:", r2) #0.843

room_val = val_set[val_set['rooms'].isna()]
room_val.drop(['rent','rooms','sell_price'],axis=1,inplace=True)
room_val['rooms'] = best_model_room.predict(room_val)
for i in room_val.index:
    val_set.loc[i,'rooms'] = room_val.loc[i,'rooms']

# to fill nas in rent
non_rent = clean_set.drop(["rent", 'sell_price'], axis=1)
rent = clean_set["rent"]
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(non_rent, rent, test_size=0.2, random_state=36)

# Define the XGBoost model
xgb_model = XGBRegressor()

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3]
    ,"enable_categorical":[True,True,True]
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

# Fit the model to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Make predictions on the test set using the best model
best_model_rent = grid_search.best_estimator_
predictions = best_model_rent.predict(X_test)
# Evaluate the model
r2 = r2_score(y_test, predictions)
print("Test R2:", r2) #0.874

rent_val = val_set[val_set['rent'].isna()]
rent_val.drop(['rent','sell_price'],axis=1,inplace=True)
rent_val['rent'] = best_model_rent.predict(rent_val)
for i in rent_val.index:
    val_set.loc[i,'rent'] = rent_val.loc[i,'rent']
#%%
#Models to predict the sell prices
#The preparations
#Firstly we use StratfiedKFold to reduce overfitting, especially effective to unbalanced dataset
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
#Get X and Y
non_sp = clean_set.drop(['sell_price'], axis=1)
sps = clean_set['sell_price']
encoded_df = pd.get_dummies(non_sp, columns=['legal_district', 'object_type', 'Period', 'association_tax_liability'], prefix='Category')
#Get the scaled X and Y
scaler = StandardScaler()
encoded_clean = pd.get_dummies(clean_set, columns=['legal_district', 'object_type', 'Period', 'association_tax_liability'], prefix='Category')
clean_set_scaled = scaler.fit_transform(encoded_clean)
clean_set_scaled = pd.DataFrame(clean_set_scaled, columns=encoded_clean.columns)
non_sp_scaled = clean_set_scaled.drop(['sell_price'], axis=1)
sps_scaled = clean_set_scaled['sell_price']
# Define kfold and scoring
kfold = KFold(n_splits=5, shuffle=True, random_state=80)
scoring = {'r2': 'r2', 'rmse': "neg_root_mean_squared_error"}

#%%
#Linear regression: basic model
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import get_scorer_names
linear_model = LinearRegression()
scores_linear = cross_validate(linear_model, encoded_df, sps, cv=kfold, scoring=scoring)

#Score
for i in scores_linear.keys():
    print(i)
    print(scores_linear[i].mean())
'''fit_time
0.19960441589355468
score_time
0.0031158924102783203
test_r2
0.8134731457705595
test_rmse
-1187880.974673568'''

print("Mean R^2 score", np.mean(scores_linear)) #0.8135


#%%
#knn
from sklearn.neighbors import KNeighborsRegressor 
for i in range(3,8):
    knn_regressor = KNeighborsRegressor(n_neighbors=i) 
    knn_scores = cross_validate(knn_regressor, encoded_df, sps, cv=kfold, scoring=scoring)
    print(i)
    for j in knn_scores.keys():
        print(j+":")
        print(knn_scores[j].mean()) #0.879
'''3
fit_time:
0.030739736557006837
score_time:
2.7905285358428955
test_r2:
0.7378485408889506
test_rmse:
-1408354.743457914'''
#%% 
#Tree
from sklearn.tree import DecisionTreeRegressor 
decision_tree_regressor = DecisionTreeRegressor()
tree_scores = cross_validate(decision_tree_regressor,encoded_df, sps, cv=kfold, scoring=scoring)
for i in tree_scores.keys():
    print(i+":")
    print(tree_scores[i].mean()) #0.879
'''fit_time:
1.282659912109375
score_time:
0.012788724899291993
test_r2:
0.8771278909652915
test_rmse:
-963706.9998335151'''
#With tuning
param_grid = { 
    'max_depth': [None, 5, 10, 15], 
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4] 
}

decision_tree_regressor = DecisionTreeRegressor() 
grid_search = GridSearchCV(decision_tree_regressor, param_grid, cv=kfold, scoring='r2', n_jobs=-1) 
grid_search.fit(encoded_df, sps) 

print("Best Parameters:", grid_search.best_params_) #
tree_best_params = grid_search.best_params_
best_decision_tree = grid_search.best_estimator_ 
cross_eval_tree = cross_validate(best_decision_tree, encoded_df, sps, cv=kfold, scoring=scoring, return_train_score=True)
for i in cross_eval_tree.keys():
    print(i)
    print(cross_eval_tree[i].mean())
'''fit_time
0.8670158386230469
score_time
0.006224441528320313
test_r2
0.9007758587976525
train_r2
0.9555149497714037
test_rmse
-866050.4883041743
train_rmse
-580177.6507427443'''

#%%XGBoost
#without tuning
xgb_regressor = XGBRegressor()
xg_bs_scores = cross_validate(xgb_regressor, encoded_df, sps, cv=kfold, scoring=scoring)
for i in xg_bs_scores.keys():
    print(i)
    print(xg_bs_scores[i].mean())
'''fit_time
0.49274134635925293
score_time
0.014988517761230469
test_r2
0.9376132472661138
test_rmse
-686900.4032626221'''
xg_bs_scores_scaled = cross_validate(xgb_regressor, non_sp_scaled, sps_scaled, cv=kfold, scoring=scoring)
for i in xg_bs_scores_scaled.keys():
    print(i)
    print(xg_bs_scores_scaled[i].mean())
'''fit_time
0.48335986137390136
score_time
0.013869857788085938
test_r2
0.9372641987538082
test_rmse
-0.2503612383922735'''
#with tuning
param_grid_xgboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3],
    "enable_categorical":[True,True,True]
}
#without scaling
grid_search = GridSearchCV(xgb_regressor, param_grid_xgboost, cv=kfold, scoring='r2') 
grid_search.fit(encoded_df, sps)

print("Best Parameters:", grid_search.best_params_) 
XG_best_params = grid_search.best_params_
best_XG = grid_search.best_estimator_ 
cross_eval_XG = cross_validate(best_XG, encoded_df, sps, cv=kfold, scoring=scoring, return_train_score=True)
for i in cross_eval_XG.keys():
    print(i)
    print(cross_eval_XG[i].mean())
'''fit_time
0.827869176864624
score_time
0.01770939826965332
test_r2
0.938273004979399
train_r2
0.9594862167348959
test_rmse
-683204.805750713
train_rmse
-553666.6991521048'''
#withscaling
xgb_regressor = XGBRegressor()
grid_search = GridSearchCV(xgb_regressor, param_grid_xgboost, cv=kfold, scoring='r2') 
grid_search.fit(non_sp_scaled, sps_scaled) 

print("Best Parameters:", grid_search.best_params_) 
XG_best_params_scaled = grid_search.best_params_
best_XG_scaled = grid_search.best_estimator_
cross_eval_XG_scaled = cross_validate(best_XG_scaled, non_sp_scaled, sps_scaled, cv=kfold, scoring=scoring, return_train_score=True)
for i in cross_eval_XG_scaled.keys():
    print(i)
    print(cross_eval_XG_scaled[i].mean())
'''fit_time
0.8801707744598388
score_time
0.01860074996948242
test_r2
0.9381609629729342
train_r2
0.9590112526090809
test_rmse
-0.24858621298384348
train_rmse
-0.2024456822036577'''

#%%
#Robustness
# Cross Evaluation
# Decision Tree

# print(cross_eval_tree['test_r2'])
# print(cross_eval_XG['test_r2'])

# choose the columns that must be kept
columns_to_keep = ['sell_price']

# delete other lines
columns_to_drop = set(encoded_clean.columns) - set(columns_to_keep)

# random select the deleted 1-4 columns
columns_to_drop_random = np.random.choice(list(columns_to_drop), size=np.random.randint(1, 4), replace=False)

# print the chosen columns
print("the deleted columns:", columns_to_drop_random)

# delete the selected columns
encoded_robust = encoded_clean.drop(columns=columns_to_drop_random, axis=1)

#apply the primary model on the new dataset
x_robust = encoded_robust.drop(['sell_price'], axis=1)
y_robust = encoded_robust['sell_price']

xgb_robust = XGBRegressor(**XG_best_params)

cross_eval_XG_robust = cross_validate(xgb_robust, x_robust, y_robust, cv=kfold, scoring=scoring, return_train_score=True)

print("The test R2 score on different folds are", cross_eval_XG_robust['test_r2'])

feature_importances = best_XG.feature_importances_
print("Feature Importances:")
for feature, importance in zip(encoded_df.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")

#%%
# Predict the sell prices on val_set

encoded_val_set = pd.get_dummies(val_set, columns=['legal_district', 'object_type', 'Period', 'association_tax_liability'], prefix='Category')

x = encoded_val_set.drop(['sell_price'], axis=1)
y_pred = best_XG.predict(x)
y_pred = pd.Series(y_pred)
encoded_val_set['sell_price'] = y_pred
encoded_val_set['sell_price'] = best_XG.predict(x)
encoded_val_set.reset_index(inplace=True)
encoded_val_set_json = pd.DataFrame(encoded_val_set[['id','sell_price']])
encoded_val_set_json.to_json("assignment2_42441.json", orient="records", indent=2)