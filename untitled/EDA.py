import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

path = 'data/'
train_data = pd.read_csv(path + 'sales_train.csv')
test_data = pd.read_csv(path + 'test.csv')
item = pd.read_csv(path + 'items.csv')

test_shop_ids = test_data['shop_id'].unique()
test_item_ids = test_data['item_id'].unique()
print(test_item_ids.shape)
print(test_shop_ids.shape)
lk_train = train_data[train_data['shop_id'].isin(test_shop_ids)]
lk_train = lk_train[lk_train['item_id'].isin(test_item_ids)]

train_data = lk_train[['date','date_block_num','shop_id','item_id','item_price','item_cnt_day']]

del train_data['date']
train_data = train_data.groupby(['date_block_num','shop_id','item_id'],as_index=False)
train_data = train_data.agg({'item_price':['sum','mean'],'item_cnt_day':['sum','mean','count']})
train_data.columns = ['date_block_num','shop_id','item_id','item_price','item_price_mean','item_cnt','item_cnt_day_mean','transaction']



train_data['year'] = train_data['date_block_num'].apply(lambda x:(x//12)+2013)
train_data['month'] = train_data['date_block_num'].apply(lambda x:(x%12))

gp_month_mean = train_data.groupby(['month'],as_index = False)['item_cnt'].mean()
gp_month_sum = train_data.groupby(['month'],as_index = False)['item_cnt'].sum()
gp_shop_mean = train_data.groupby(['shop_id'],as_index = False)['item_cnt'].mean()
gp_shop_sum = train_data.groupby(['shop_id'],as_index = False)['item_cnt'].sum()

f,ax = plt.subplots(1,2,figsize=(22,10))
sns.lineplot(x = 'month',y='item_cnt',data = gp_month_mean,ax = ax[0]).set_title('monthly_mean')
sns.lineplot(x = 'month',y='item_cnt',data = gp_month_sum,ax=ax[1]).set_title('monthly_sum')

f,ax = plt.subplots(1,2,figsize=(22,10))
sns.barplot(x='shop_id',y='item_cnt',data= gp_shop_mean,ax = ax[0],palette='rocket').set_title('shop_mean')
sns.barplot(x='shop_id',y = 'item_cnt',data = gp_shop_sum,ax = ax[1],palette='rocket').set_title('shop_cnt')

f,ax = plt.subplots(2,1,figsize= (22,10))
sns.boxplot(train_data['item_cnt'],ax = ax[0])
sns.boxplot(train_data['item_price'],ax=ax[1])
plt.show()

train_data = train_data.query('item_cnt >=0 and item_cnt <=20 and item_price <=400000')
shop_ids = test_data['shop_id'].unique()
item_ids = test_data['item_id'].unique()
print(shop_ids.shape)
print(item_ids.shape)

empty_df =[]
for i in range(34):
    for shop in shop_ids:
        for item in item_ids:
            empty_df.append([i,shop,item])

empty_df = pd.DataFrame(empty_df,columns=['date_block_num','shop_id','item_id'])
train_data = pd.merge(empty_df,train_data,on = ['date_block_num','shop_id','item_id'],how = 'left')
train_data.fillna(0,inplace = True)

train_data.to_csv(path +'train_data.csv',index = False)