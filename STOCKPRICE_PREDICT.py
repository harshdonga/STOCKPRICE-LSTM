# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:57:34 2019

@author: Harsh
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np
import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = 20,10

dataset = pd.read_csv('C://Users/Harsh/Desktop/Trials/STOCKPRICE/NKE.csv')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

dataset['Date'] = pd.to_datetime(dataset.Date , format= '%Y-%m-%d')
dataset.index = dataset['Date']

plt.figure(figsize=(16,8))
plt.plot(dataset['Close']) 
plt.legend('Nike Closing Prices')

data = dataset.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(dataset)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data.index = new_data.Date
new_data.drop('Date',axis=1,inplace=True)

fulldata = new_data.values

trainset = fulldata[:7744,:]
testset = fulldata[7744:,:]

trainset_scaled = scaler.fit_transform(trainset)

x_train , y_train = [], []
for i in range(60,len(trainset_scaled)):
    x_train.append(trainset_scaled[i-60:i, 0])
    y_train.append(trainset_scaled[i,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense, LSTM , Dropout


model = Sequential()
model.add(LSTM(units=50,return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error' , optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=200, verbose=1)

dataset_test = testset
real_stock_price = dataset_test.iloc[:, 1:2].values

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Nike Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Nike Stock Price')
plt.legend()
plt.show()