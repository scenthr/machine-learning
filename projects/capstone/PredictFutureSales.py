# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:29:28 2019

@author: s.pavkovic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from itertools import product
import pickle

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from xgboost import plot_importance

os.chdir('C:/Users/s.pavkovic/Desktop/Python/Predict Future Sales/')

def df_stats(df):
    print("----------Top-5- Record----------")
    print(df.head())
    print("-----------Information-----------")
    print(df.info())
    print("-----------Data Types-----------")
    print(df.dtypes)
    print("----------Missing value-----------")
    print(df.isnull().sum())
    print("----------Null value-----------")
    print(df.isna().sum())
    print("----------Shape of Data----------")
    print(df.shape)
    print("----------Potential Duplicates----------")
    print(df.duplicated().sum())

def elapsed_time(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


df_item_categories = pd.read_csv('competitive-data-science-predict-future-sales/item_categories.csv')  # 84 
df_items = pd.read_csv('competitive-data-science-predict-future-sales/items.csv') # 22170
df_shops = pd.read_csv('competitive-data-science-predict-future-sales/shops.csv') # 60

df = pd.read_csv('competitive-data-science-predict-future-sales/sales_train_v2.csv')  # 2935849
df_test = pd.read_csv('competitive-data-science-predict-future-sales/test.csv').set_index('ID') # 214200

# EDDA
# Data inspection

# Although there is an outlier in the data i am not convinced that it should be removed from the dataset yet
df.boxplot(column=['item_cnt_day']);
plt.savefig('Boxplot_item_cnt_day.png')
df[df['item_id']==11373].boxplot(column=['item_cnt_day']);
df[df['item_id']==11373].boxplot(column=['item_price']);
df[df['item_cnt_day']>2000].T
df_test[df_test['item_id']==11373]

df.boxplot(column=['item_price']);
plt.savefig('Boxplot_item_price.png')
df[df['item_price']>50000].T
df_test[df_test['item_id']==6066]
df_test[df_test['item_id']==13199]
df_test[df_test['item_id']==11365]

# Some item prices are below 0, but not influence as not in the test df
df[df['item_price']<0].T
df_test[df_test['item_id']==2973]

# Item counts per day and revenue

#t = pd.concat([df['date'], df['item_cnt_day']], axis=1)
#t['date'] = pd.to_datetime(t['date'])
#t.plot()

t = df
t = t.set_index('date')
t.index = pd.to_datetime(t.index, format='%d.%m.%Y')
t['revenue'] = t['item_cnt_day'] * t['item_price']
t = t.sort_index()

dt = t['item_cnt_day'].resample('M').sum()
dt.plot(legend=True, label="Number of products sold")
plt.savefig('Product sales over period.png')

dtt = t['revenue'].resample('M').sum()
dtt.plot(legend=True, label="Monthly revenue")
plt.savefig('Revenue over period.png')


df_items['item_category_id'].value_counts()[:20].plot(kind='bar',legend=True, label="Items per category")
plt.savefig('Item categories and item counts.png')


# Looking into the shops there seem to be some duplicates [thanks to Denis Larionov insights]
# 0 = 57, 1 = 58, 10 = 11
df_shops
df.loc[df.shop_id == 0, 'shop_id'] = 57
df_test.loc[df_test.shop_id == 0, 'shop_id'] = 57

df.loc[df.shop_id == 1, 'shop_id'] = 58
df_test.loc[df_test.shop_id == 1, 'shop_id'] = 58

df.loc[df.shop_id == 10, 'shop_id'] = 11
df_test.loc[df_test.shop_id == 10, 'shop_id'] = 11

# comparing the test and train datasets
# there are 363 items in 15246 observations that are not present in the data

len(df_test[~df_test['item_id'].isin(df['item_id'])].item_id.unique())  # 363 missing
len(df_test.item_id.unique())  # 5100 diff items 
len(df.item_id.unique())  # 21807 diff items

# these missing items are still present in the df_items table so we know their
# category
### We could use the categories to create some data for these items?
df_items.loc[df_items.item_id == 8969]


# TRANSFORMING
# Transformin both test and train sets so that we get the similar format

## Train set - Creating a cohort for each month and each shop and item

matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    sales = df[df.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)


# Adding a column revenue
df['revenue'] = df['item_price'] *  df['item_cnt_day']

# Group the number of items sold and add to matrix

items = df.groupby(['date_block_num','shop_id','item_id']).item_cnt_day.sum()
items = items.reset_index()
items.columns = ['date_block_num','shop_id','item_id','item_cnt_month']

cols = ['date_block_num','shop_id','item_id']
matrix = pd.merge(matrix, items, on=cols, how='left')   
matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0)
matrix['item_cnt_month'] = matrix['item_cnt_month'].clip(0,20)
matrix['item_cnt_month'] = matrix['item_cnt_month'].astype(np.float16)

## adding the test information to matrix as 34. month - the one that we want to predict
## the data for

df_test['date_block_num'] = 34
df_test['date_block_num'] = df_test['date_block_num'].astype(np.int8)
df_test['shop_id'] = df_test['shop_id'].astype(np.int8)
df_test['item_id'] = df_test['item_id'].astype(np.int16)

matrix = pd.concat([matrix, df_test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) 

# Adding the item category id and item name to the matrix
matrix = pd.merge(matrix, df_items, on=['item_id'], how='left')
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix.drop(['item_name'], axis=1, inplace=True)

def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

# Added the lags for the feature we are trying to predict
matrix = lag_feature(matrix, [1,2,3,6,9,12], 'item_cnt_month')


def add_group_mean_lag(df,groupby,agg_feature_name,lags,new_feature_name):
#    df = matrix
#    groupby = ['date_block_num']
#    agg_feature_name = 'item_cnt_month'
#    lags = [1]    
#    new_feature_name = 'date_avg_item_cnt'
       
    ts = time.time()
    
    loc = df.groupby(groupby).agg({agg_feature_name:['mean']})
    loc.columns = [new_feature_name]
    loc.reset_index(inplace=True)
    
    df = pd.merge(df,loc,on=groupby,how='left')
    df[new_feature_name] = df[new_feature_name].astype(np.float16)
    df = lag_feature(df, lags, new_feature_name)
    df.drop([new_feature_name], axis=1, inplace=True)    
    
    print('Elapsed: {}'.format(time.time() - ts))
        
    return(df)
    
# Lag for the last month mean average items sold
matrix = add_group_mean_lag(matrix,
                            ['date_block_num'],
                            'item_cnt_month',
                            [1],
                            'block_avg_item_cnt')
# lags for block and item
matrix = add_group_mean_lag(matrix,
                            ['date_block_num', 'item_id']
                            ,'item_cnt_month',
                            [1,2,3,6,9,12],
                            'block_item_avg_item_cnt')

# lags for block and shop
matrix = add_group_mean_lag(matrix
                            ,['date_block_num', 'shop_id']
                            ,'item_cnt_month'
                            ,[1,2,3,6,9,12]
                            ,'block_shop_avg_item_cnt')

# lags for block and item category
matrix = add_group_mean_lag(matrix
                            ,['date_block_num', 'item_category_id']
                            ,'item_cnt_month'
                            ,[1,12]
                            ,'block_cat_avg_item_cnt')

# lags for block and item category
matrix = add_group_mean_lag(matrix
                            ,['date_block_num', 'shop_id','item_category_id']
                            ,'item_cnt_month'
                            ,[1,11]
                            ,'block_shop_cat_avg_item_cnt')

# lags for block and item category
matrix = add_group_mean_lag(matrix
                            ,['date_block_num', 'shop_id','item_category_id']
                            ,'item_cnt_month'
                            ,[1,11]
                            ,'block_shop_cat_avg_item_cnt')

# Additional usefull features
matrix['month'] = matrix['date_block_num'] % 12

# Add period when the item was last sold (per shop)
t = matrix[(matrix['date_block_num']!=34)].groupby(['shop_id','item_id']).agg({'date_block_num':['max']})
t.columns = ['shop_item_last_purchased']
t.reset_index(inplace=True)

matrix = pd.merge(matrix,t,on=['shop_id','item_id'],how='left')
matrix['shop_item_last_purchased'] = matrix['shop_item_last_purchased'].astype(np.float16)
matrix['shop_item_last_purchased'].fillna(-1, inplace=True) 


# Add period when the item was last sold
t = matrix[matrix['date_block_num']!=34].groupby(['item_id']).agg({'date_block_num':['max']})
t.columns = ['item_last_purchased']
t.reset_index(inplace=True)

matrix = pd.merge(matrix,t,on=['item_id'],how='left')
matrix['item_last_purchased'] = matrix['item_last_purchased'].astype(np.float16)
matrix['item_last_purchased'].fillna(-1, inplace=True) 

# Add period when the item was first sold (per shop)
t = matrix[(matrix['date_block_num']!=34)].groupby(['shop_id','item_id']).agg({'date_block_num':['min']})
t.columns = ['shop_item_first_purchased']
t.reset_index(inplace=True)

matrix = pd.merge(matrix,t,on=['shop_id','item_id'],how='left')
matrix['shop_item_first_purchased'] = matrix['shop_item_first_purchased'].astype(np.float16)
matrix['shop_item_first_purchased'].fillna(-1, inplace=True) 


# Add period when the item was first sold
t = matrix[matrix['date_block_num']!=34].groupby(['item_id']).agg({'date_block_num':['min']})
t.columns = ['item_first_purchased']
t.reset_index(inplace=True)

matrix = pd.merge(matrix,t,on=['item_id'],how='left')
matrix['item_first_purchased'] = matrix['item_first_purchased'].astype(np.float16)
matrix['item_first_purchased'].fillna(-1, inplace=True) 

# FINAL PREPARATION - fill the zeros for the missing values
matrix.fillna(0,inplace=True)

# we drop the first 12 months as we have used lags of max 12 months
matrix_full = matrix
matrix = matrix[matrix.date_block_num > 11]
matrix.to_pickle('matrix.pkl')


## Work with the final matrix
df = pd.read_pickle('matrix.pkl')



# MODELLING

############ Linear Regression

X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = df[df.date_block_num < 33]['item_cnt_month']
X_valid = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = df[df.date_block_num == 33]['item_cnt_month']
X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)

# Scaling not great on the data

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_valid = scaler.fit_transform(X_valid)
# X_test = scaler.fit_transform(X_test)

model = linear_model.LinearRegression()
model.fit(
    X_train, 
    Y_train    
    )

Y_pred = model.predict(X_valid).clip(0,20)
# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_valid, Y_pred))

print("Root mean squared error: %.2f"
      % mean_squared_error(Y_valid, Y_pred)**(1/2))

Y_test = model.predict(X_test).clip(0, 20)

# Prepare for submission

submission = pd.DataFrame({
    "ID": df_test.index, 
    "item_cnt_month": Y_test
})

submission.to_csv('lg_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('lg_train.pickle', 'wb'))
pickle.dump(Y_test, open('lg_test.pickle', 'wb'))


############ Lasso
model = linear_model.Lasso(alpha=0.5)
model.fit(
    X_train, 
    Y_train    
    )

Y_pred = model.predict(X_valid).clip(0,20)
# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_valid, Y_pred))

print("Root mean squared error: %.2f"
      % mean_squared_error(Y_valid, Y_pred)**(1/2))

Y_test = model.predict(X_test).clip(0, 20)

# Prepare for submission

submission = pd.DataFrame({
    "ID": df_test.index, 
    "item_cnt_month": Y_test
})

submission.to_csv('lasso_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('lasso_train.pickle', 'wb'))
pickle.dump(Y_test, open('lasso_test.pickle', 'wb'))


############ Ridge Regression

model = linear_model.Ridge(alpha=100)
model.fit(
    X_train, 
    Y_train    
    )

Y_pred = model.predict(X_valid).clip(0,20)
# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_valid, Y_pred))

print("Root mean squared error: %.2f"
      % mean_squared_error(Y_valid, Y_pred)**(1/2))

Y_test = model.predict(X_test).clip(0, 20)

# Prepare for submission

submission = pd.DataFrame({
    "ID": df_test.index, 
    "item_cnt_month": Y_test
})

submission.to_csv('ridge_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('ridge_train.pickle', 'wb'))
pickle.dump(Y_test, open('ridge_test.pickle', 'wb'))



########### XGBoost

X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = df[df.date_block_num < 33]['item_cnt_month']
X_valid = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = df[df.date_block_num == 33]['item_cnt_month']
X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42
    )

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

# Prepare for submission

submission = pd.DataFrame({
    "ID": df_test.index, 
    "item_cnt_month": Y_test
})

submission.to_csv('xgb_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))

# important features
fig, ax = plt.subplots(1,1,figsize=(10,16))
plot_importance(booster=model, ax=ax)
plt.savefig('XGBoost_Feature_Importances.jpg')


########### DNN with Keras
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(32, input_dim=32, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define the model
def deeper_model():
	# create model
	model = Sequential()
	model.add(Dense(32, input_dim=32, kernel_initializer='normal', activation='relu'))
	model.add(Dense(16, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define wider model
def wider_model():
	# create model
    model = Sequential()
    #Input layer
    model.add(Dense(48, input_dim=32, kernel_initializer='normal', activation='relu'))
    # Hidden layer
    model.add(Dense(96, kernel_initializer='normal',activation='relu'))
    # Output layer
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

@elapsed_time
def run_model(model, epochs, kfold_splits):
    seed = 7
    numpy
    numpy.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=epochs, batch_size=5, verbose=1)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=kfold_splits, random_state=seed)
    results = cross_val_score(pipeline, X_train_sample, Y_train_sample, cv=kfold)
    print("Model: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate baseline model with standardized dataset    
run_model(baseline_model, 50, 8)    
run_model(deeper_model, 50, 8)
run_model(wider_model, 50, 8)
    
# sampling as not enough processing power
X_train_sample = X_train.sample(n=100000, random_state=42)
Y_train_sample = Y_train[X_train_sample.index]

model = wider_model()
model.summary()

model.fit(X_train_sample,Y_train_sample, epochs=30, batch_size=16, validation_split = 0.2)

Y_pred = model.predict(X_valid).clip(0, 20)
MSE = mean_squared_error(Y_valid , Y_pred)
MSE
RMSE = MSE**(1/2)

Y_test = model.predict(X_test).clip(0, 20)

# Prepare for submission

submission = pd.DataFrame({
    "ID": df_test.index, 
    "item_cnt_month": Y_test[:,0]
})

submission.to_csv('DNN_wide_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('DNN_wide_train.pickle', 'wb'))
pickle.dump(Y_test, open('DNN_wide.pickle', 'wb'))











