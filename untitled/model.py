import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

path = 'data/'
train_data = pd.read_csv(path+ 'train_set.csv')
validation_data = pd.read_csv(path+'validation_set.csv')
test_data = pd.read_csv(path+'test_set.csv')
tt = pd.read_csv(path+'test.csv')

print(train_data.head().append(train_data.tail()))
train_data.fillna(0,inplace=True)
validation_data.fillna(0,inplace=True)
x_train = train_data.drop(['date_block_num','item_cnt_next_month'],axis=1)
y_train = train_data['item_cnt_next_month'].astype(int)
x_validation = validation_data.drop(['date_block_num','item_cnt_next_month'],axis=1)
y_validation = validation_data['item_cnt_next_month'].astype(int)

int_features = ['shop_id','item_id','year','month']
x_train[int_features] = x_train[int_features].astype('int32')
x_validation[int_features] = x_validation[int_features].astype('int32')

x_test = test_data[x_train.columns]
x_test[int_features] = x_test[int_features].astype('int32')

#catboost
cat_features=['shop_id','item_id','year','month']
cat_model = CatBoostRegressor(
    iterations=500,
    max_ctr_complexity=4,
    random_seed=2020,
    depth=6,
    verbose=100,
    od_type='Iter',
    od_wait=25,
    border_count=254,
    eval_metric='RMSE',
    task_type='GPU',
    loss_function='RMSE',
    l2_leaf_reg=3
)

cat_model.fit(x_train,y_train,cat_features=cat_features,eval_set=(x_validation,y_validation),plot=True)
final_cat_model = cat_model
cat_y_tra_pred = final_cat_model.predict(x_train)
cat_y_val_pred = final_cat_model.predict(x_validation)
cat_tra_score = np.sqrt(mean_absolute_error(y_train,cat_y_tra_pred))
cat_val_score = np.sqrt(mean_absolute_error(y_validation,cat_y_val_pred))
print('********cat_tra_score:{:<8.8f}\n'.format(cat_tra_score))
print('********cat_val_score:{:<8.8f}\n'.format(cat_val_score))
cat_pred = final_cat_model.predict(x_test)
with open(path+'res.txt','a') as f:
    f.write('cat_tra_score{} ============= cat_val_score{}\n'.format(cat_tra_score,cat_val_score))


#xgb
query_features = ['shop_id','item_id','year','month']
xgb_features = [col for col in x_train.columns if col not in query_features]
xgb_train = x_train[xgb_features]
xgb_val = x_validation[xgb_features]
xgb_test = x_test[xgb_features]

xgb_model = XGBRegressor(
    max_depth=8,
    n_estimators=500,
    min_child_weight=1000,
    colsample_bytree=0.7,
    subsample=0.7,
    eta = 0.3,
    seed=2020
)

xgb_model.fit(xgb_train,y_train,eval_metric='rmse',eval_set=[(xgb_train,y_train),(xgb_val,y_validation)],verbose=True,early_stopping_rounds=20)
final_xgb_model = xgb_model
xgb_y_tra_pred = final_xgb_model.predict(xgb_train)
xgb_y_val_pred = final_xgb_model.predict(xgb_val)
xgb_tra_score = np.sqrt(mean_absolute_error(y_train,xgb_y_tra_pred))
xgb_val_score = np.sqrt(mean_absolute_error(y_validation,xgb_y_val_pred))
print('********xgb_tra_score:{:<8.8f}\n'.format(xgb_tra_score))
print('********xgb_val_score:{:<8.8f}\n'.format(xgb_val_score))
xgb_pred = final_xgb_model.predict(xgb_test)
with open(path+'res.txt','a') as f:
    f.write('**********{}********{}\n'.format(xgb_tra_score,xgb_val_score))

#lgb
param = {
'num_leaves': 80,
'min_data_in_leaf': 40,
'objective':'regression',
'max_depth': -1,
'learning_rate': 0.1,
"min_child_samples": 30,
"boosting": "gbdt",
"feature_fraction": 0.9,
"bagging_freq": 2,
"bagging_fraction": 0.9,
"bagging_seed": 2029,
"metric": 'rmse',
"lambda_l1": 0.1,
"lambda_l2": 0.2,
"verbosity": -1}

lgb_model = LGBMRegressor(**param)
lgb_model.fit(xgb_train,y_train,eval_set=(xgb_val,y_validation),verbose=True,early_stopping_rounds=20,eval_metric='rmse')
final_lgb_model = lgb_model
lgb_y_tra_pred = final_lgb_model.predict(xgb_train)
lgb_y_val_pred = final_lgb_model.predict(xgb_val)
lgb_tra_score = np.sqrt(mean_absolute_error(y_train,lgb_y_tra_pred))
lgb_val_score = np.sqrt(mean_absolute_error(y_validation,lgb_y_val_pred))
print('********lgb_tra_score:{:<8.8f}\n'.format(lgb_tra_score))
print('********lgb_val_score:{:<8.8f}\n'.format(lgb_val_score))
lgb_pred = final_lgb_model.predict(xgb_test)
with open(path+'res.txt','a') as f:
    f.write('********{}*******{}\n'.format(lgb_tra_score,lgb_val_score))

#stacking
train_stack = np.vstack([cat_y_val_pred,xgb_y_val_pred,lgb_y_val_pred]).transpose()
test_stack = np.vstack([cat_pred,xgb_pred,lgb_pred]).transpose()

clf = BayesianRidge()
clf.fit(train_stack,y_validation)
final_pred = clf.predict(test_stack)

print(final_pred.shape)
print(tt.shape)


#output
sub = pd.DataFrame(tt['ID'],columns=['ID'])
sub['item_cnt_month'] = final_pred.clip(0.,20.)
sub.to_csv(path+'submission.csv',index=False)
print(sub.head(10))