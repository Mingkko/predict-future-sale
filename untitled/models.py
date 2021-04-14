import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
import math


path = 'data/'
train_data = pd.read_csv(path + 'train_data.csv')
test_data = pd.read_csv(path + 'test.csv')
print(train_data.head())

del train_data['date']

train_data = train_data[np.abs(train_data['item_price']-train_data['item_price'].mean())<=(3*train_data['item_price'].std())]
train_data = train_data[np.abs(train_data['item_cnt_day']-train_data['item_cnt_day'].mean())<=(3*train_data['item_cnt_day'].std())]

test_shop_ids = test_data['shop_id'].unique()
test_item_ids = test_data['item_id'].unique()

lktrian = train_data[train_data['shop_id'].isin(test_shop_ids)]

lktrian = lktrian[lktrian['item_id'].isin(test_item_ids)]

train_data = lktrian[['item_id','shop_id','item_price','item_cnt_day','date_block_num']]

label = ['item_cnt_day']

def get_labels(train):
    y = np.array(train[label])

    return y


def get_x(train):
    cols = train.columns
    features = [col for col in cols if col not in label]
    x = np.array(train[features])

    return x


rf = RandomForestClassifier()
knn = KNeighborsClassifier()
lr = LinearRegression()
svc = SVR()
gbdt = GradientBoostingClassifier()

models = [knn,lr,rf,gbdt]
param_grid_list = [
    #knn
    [{
        'n_neighbors':[5,10,15,30],
        'leaf_size':[10,20,30,40]
    }],

    #lr
    [{
        'n_jobs':[5]
    }],

    #rf
    [{
        'n_estimators':[200,250,300,350],
    }],

    #gbdt
    [{
        'learning_rate':[0.1,0.5],
        'n_estimators':[150,200,250],
        'min_samples_split':[500,1000,2000],
        'min_samples_leaf':[60,100],
        'subsample':[0.8,1],
        'max_features':['sqrt'],
        'max_depth':[5,7]
    }]

]

# folds = KFold(n_splits=10,shuffle=True,random_state=2020)
# x_train = get_x(train_data)
# y_train = get_labels(train_data)
# kfold = folds.split(x_train,y_train)
# for tra_idx,val_idx in kfold:
#     train_part = x_train[tra_idx]
#     valid_part = x_train[val_idx]
#     train_part_y = y_train[tra_idx]
#     valid_part_y = y_train[val_idx]


x_train = get_x(train_data)
y_train = get_labels(train_data)

train_x,test_x,train_y,test_y = train_test_split(x_train,y_train,test_size=0.3,random_state=2020)

print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)
train_part = train_x[:84541]
valid_part = test_x[:36232]
train_part_y = train_y[:84541]
valid_part_y = test_y[:36232]


def my_loss(x,y):
    return math.sqrt(mean_squared_error(x,y))


score = make_scorer(my_loss,greater_is_better=False)

for i,model in enumerate(models):

    grid_search = GridSearchCV(model,param_grid_list[i],cv=5,verbose=100,n_jobs=5,scoring=score)
    grid_search.fit(train_part,train_part_y.flatten())
    print(grid_search.best_estimator_)
    final_model = grid_search.best_estimator_
    pred = final_model.predict(valid_part)
    print('************rmse score{:<8.8f}\n'.format(math.sqrt(mean_squared_error(pred,valid_part_y))))
    with open(path+'res.txt','a') as f:
        f.write('{}****score{:<8.8f}\n'.format(i,math.sqrt(mean_squared_error(pred,valid_part_y))))