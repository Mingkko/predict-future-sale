import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

path = 'data/'

train_data = pd.read_csv(path + 'train_data.csv')



train_data['item_cnt_next_month'] = train_data.sort_values('date_block_num').groupby(['shop_id','item_id'])['item_cnt'].shift(-1)
train_data['item_price_unit'] = train_data['item_price'] // train_data['item_cnt']
train_data['item_price_unit'].fillna(0,inplace=True)

gp_item_price = train_data.sort_values('date_block_num').groupby(['item_id'],as_index=False).agg({'item_price':[np.min,np.max]})
gp_item_price.columns = ['item_id','min_item_price','max_item_price']

train_data = pd.merge(train_data,gp_item_price,on = 'item_id', how = 'left')

train_data['price_increase'] = train_data['item_price'] - train_data['min_item_price']
train_data['price_decrease'] = train_data['max_item_price'] - train_data['item_price']

print(train_data.head())

f_min = lambda x:x.rolling(window=3,min_periods=1).min()
f_max = lambda x:x.rolling(window=3,min_periods=1).max()
f_mean = lambda x:x.rolling(window=3,min_periods=1).mean()
f_std = lambda x:x.rolling(window=3,min_periods=1).std()

function_list = [f_min,f_max,f_mean,f_std]
function_name = ['min','max','mean','std']

for i in range(len(function_list)):
    train_data[('item_cnt_%s'%function_name[i])] = train_data.sort_values('date_block_num').groupby(['shop_id','item_id'])['item_cnt'].apply(function_list[i])

train_data['item_cnt_std'].fillna(0,inplace = True)

lag_list = [1,2,3]
print(train_data.head())

for lag in lag_list:
    ft_name = ('item_cnt_shift{}'.format(lag))
    train_data[ft_name] = train_data.sort_values('date_block_num').groupby(['shop_id','item_id'])['item_cnt'].shift(lag)
    train_data[ft_name].fillna(0,inplace = True)


train_data['item_trend'] = train_data['item_cnt']

for lag in lag_list:
    ft_name = ('item_cnt_shift{}'.format(lag))
    train_data['item_trend'] -= train_data[ft_name]

train_data['item_trend'] /=len(lag_list)

train_set = train_data.query('date_block_num<28 and date_block_num>=3').copy()
validation_set = train_data.query('date_block_num >=28 and date_block_num <33').copy()
test_set = train_data[train_data['date_block_num'] == 33].copy()
print(test_set.shape)

train_set.dropna(subset = ['item_cnt_next_month'],inplace = True)
validation_set.dropna(subset =['item_cnt_next_month'],inplace = True)
train_set.dropna(inplace = True)
validation_set.dropna(inplace =True)

#mean encoding
train_set['shop_mean'] = train_set.groupby('shop_id')['item_cnt_next_month'].transform('mean').values
train_set['item_mean'] = train_set.groupby('item_id')['item_cnt_next_month'].transform('mean').values
train_set['shop_item_mean'] = train_set.groupby(['shop_id','item_id'])['item_cnt_next_month'].transform('mean').values
train_set['year_mean'] = train_set.groupby('year')['item_cnt_next_month'].transform('mean').values
train_set['month_mean'] = train_set.groupby('month')['item_cnt_next_month'].transform('mean').values

validation_set['shop_mean'] = validation_set.groupby('shop_id')['item_cnt_next_month'].transform('mean').values
validation_set['item_mean'] = validation_set.groupby('item_id')['item_cnt_next_month'].transform(np.mean).values
validation_set['shop_item_mean'] = validation_set.groupby(['shop_id','item_id'])['item_cnt_next_month'].transform('mean').values
validation_set['year_mean'] = validation_set.groupby('year')['item_cnt_next_month'].transform(np.mean).values
validation_set['month_mean'] = validation_set.groupby('month')['item_cnt_next_month'].transform(np.mean).values


print(test_set.head())

latest_records = pd.concat([train_set,validation_set]).drop_duplicates(subset=['shop_id','item_id'],keep='last')
test_set = pd.merge(test_set,latest_records,on=['shop_id','item_id'],how='left',suffixes=['','_'])
test_set['year'] = 2015
test_set['month'] = 9
test_set.fillna(test_set.mean(),inplace = True)
print(test_set.shape)


train_set.to_csv(path+'train_set.csv',index = False)
validation_set.to_csv(path+'validation_set.csv',index = False)
test_set.to_csv(path+'test_set.csv',index = False)

