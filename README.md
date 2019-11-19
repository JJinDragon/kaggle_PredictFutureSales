"# kaggle_PredictFutureSales" 

- Kaggle_PredictFutureSales 에 대한 제 나름대로의 해결방법입니다.
- Keras 통해 해결해보았습니다.
- 감사합니다.

● 2019. 11. 20 06:04 기준
● 2184th
● 1.01893 score
● 제출 5회

# 1.기본 라이브러리 import
import time
start= time.time() #컴파일 소요 시간을 확인하기 위함.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Dropout
from keras.layers import RepeatVector, TimeDistributed,Flatten, Activation,ThresholdedReLU, Embedding
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# 2.데이터 불러오기 및 전처리
train = pd.read_csv('data/sales_train_v2.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('data/shops.csv')
items = pd.read_csv('data/items.csv')
item_categories = pd.read_csv('data/item_categories.csv') #cats
test = pd.read_csv('data/test.csv') #val
#sample_submission=pd.read_csv('sample_submission.csv')
print(train.head())

print(max(train['date']),min(train['date']))

print(len(train),train.isnull().sum())

print(shops.head())

print(items.head())

print(item_categories.head())

print(test.head())

import os
print(os.listdir("./"))

# date 부분은 시계열 데이터로, 년도와 월로 변경 후 시계열 데이터로 변경한다.
df_cnt= train.groupby([train.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df_cnt = df_cnt[['date','item_id','shop_id','item_cnt_day']]
df_cnt['item_cnt_day'].clip(0.,20.,inplace=True)

#피벗 테이블로 변경
df_cnt=df_cnt.pivot_table(index=['item_id','shop_id'],columns='date',
                            values='item_cnt_day',fill_value=0).reset_index()

print(df_cnt.head())

# test 데이터 역시 생성한 피벗 테이블과 merge한다.
test_df_cnt = pd.merge(test,df_cnt,on=['item_id','shop_id'], how='left')
print(test_df_cnt.isnull().sum())

test_df_cnt=test_df_cnt.fillna(0)
print(test_df_cnt.head())

#id, item_id, shop_id 칼럼 제거
test_df_cnt = test_df_cnt.drop(labels=['ID','item_id','shop_id'],axis=1)

# 값의 범위가 큰 item_price는 MinMaxScaler로 값 변경
scaler = MinMaxScaler(feature_range=(0, 1))
train["item_price"] = scaler.fit_transform(train["item_price"].values.reshape(-1,1))

# item_price 값으로 피벗 테이블 생성
df_date = train.groupby([train.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
df_date = df_date[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()

#테스트 데이터와 item_price 값을 가지는 df_date를 merge
print(df_date.head())

sales = pd.merge(test,df_date,on=['item_id','shop_id'], how='left').fillna(0)
sales = sales.drop(labels=['ID','item_id','shop_id'],axis=1)

# 3. 학습 데이터 생성
y_train = test_df_cnt['2015-10'] #data-> 2013.01.01 ~ 2015.10.31 중 마지막 2015.10 predict로 학습
x_sales = test_df_cnt.drop(labels=['2015-10'],axis=1) #y train 부분 제거
x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1)) #행렬 구조 변경
x_prices = sales.drop(labels=['2015-10'],axis=1) #y_train 부 제거
x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))

X = np.append(x_sales,x_prices,axis=2) #학습할 X 
y = y_train.values.reshape((214200, 1)) #학습할 y

print("Training Predictor Shape: ",X.shape)
print("Training Predictee Shape: ",y.shape)
#del y_train, x_sales; gc.collect()

test_df_cnt = test_df_cnt.drop(labels=['2013-01'],axis=1)
x_test_sales = test_df_cnt.values.reshape((test_df_cnt.shape[0], test_df_cnt.shape[1], 1))
x_test_prices = sales.drop(labels=['2013-01'],axis=1)
x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))

test_df_cnt = np.append(x_test_sales,x_test_prices,axis=2)
#del x_test_sales,x_test_prices, price; gc.collect()
print("Test Predictor Shape: ",test.shape)

# 4. 모델링
model_lstm = Sequential()
model_lstm.add(LSTM(16, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model_lstm.add(Dropout(0.5))
model_lstm.add(LSTM(32))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam", loss='mse', metrics=["mse"])
print(model_lstm.summary())

# batch_size=128, epochs=10
LSTM_PARAM = {"batch_size":128,"verbose":2,"epochs":10}
modelstart = time.time()	

#5. 학습
#Validation data로 학습
if True:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=False)
    print("X Train Shape: ",X_train.shape)
    print("X Valid Shape: ",X_valid.shape)
    print("y Train Shape: ",y_train.shape)
    print("y Valid Shape: ",y_valid.shape)

    callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=3,mode='auto')]
    #keras에서 자동으로 학습이 불필요할 시 종료해주는 명령어
    
    hist = model_lstm.fit(X_train, y_train,validation_data=(X_valid, y_valid),
                          callbacks=callbacks_list,**LSTM_PARAM)
    pred = model_lstm.predict(test_df_cnt)
    best = np.argmin(hist.history["val_loss"])
    print("Optimal Epoch: {}",best)
    print("Train Score: {}, Validation Score: {}".format(hist.history["loss"][best],hist.history["val_loss"][best]))

# 6.제출양식 구성
submission = pd.DataFrame(pred,columns=['item_cnt_month'])
submission.to_csv('submission.csv',index_label='ID')

print("\nModel Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - start)/60))