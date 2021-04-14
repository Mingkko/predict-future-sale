import numpy as np
import pandas as pd

temp = pd.read_csv('data/test.csv')
print(temp.head(10))
print(temp.shape)
tt = pd.read_csv('data/train_data.csv')

tt = tt[tt['date_block_num'] == 33]
print(tt.shape)