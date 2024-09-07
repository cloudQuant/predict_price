import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten

from pylab import plt, mpl

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
os.environ['PYTHONHASHSEED'] = '0'

data = pd.read_pickle('../data/BTC.pkl.gz', compression='gzip')
data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
print(data.head())
print(data.info())

lags = 5
features = ['close', 'r', 'd', 'sma', 'min', 'max', 'mom', 'vol']


def add_lags(data, symbol, lags, window=20, features=features):
    cols = []
    df = data.copy()
    df.dropna(inplace=True)
    df['r'] = np.log(df['close'] / df['close'].shift(1))
    df['sma'] = df[symbol].rolling(window).mean()
    df['min'] = df[symbol].rolling(window).min()
    df['max'] = df[symbol].rolling(window).max()
    df['mom'] = df['r'].rolling(window).mean()
    df['vol'] = df['r'].rolling(window).std()
    df.dropna(inplace=True)
    df['d'] = np.where(df['r'] > 0, 1, 0)
    for f in features:
        for lag in range(1, lags + 1):
            col = f'{f}_lag_{lag}'
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols


data, cols = add_lags(data, 'close', lags, window=20, features=features)
split = int(len(data) * 0.8)
train = data.iloc[:split].copy()
mu, std = train[cols].mean(), train[cols].std()
train[cols] = (train[cols] - mu) / std
test = data.iloc[split:].copy()
test[cols] = (test[cols] - mu) / std


# Using TensorFlow backend.
def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds()
model = Sequential()
model.add(Conv1D(filters=96, kernel_size=5, activation='relu',
                 input_shape=(len(cols), 1)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(np.atleast_3d(train[cols]), train['d'],
          epochs=60, batch_size=48, verbose=False,
          validation_split=0.15, shuffle=False)

res = pd.DataFrame(model.history.history)
res.tail(3)
res.plot(figsize=(10, 6))

model.evaluate(np.atleast_3d(test[cols]), test['d'])

test['p'] = np.where(model.predict(np.atleast_3d(test[cols])) > 0.5, 1, 0)
test['p'] = np.where(test['p'] > 0, 1, -1)
test['p'].value_counts()

(test['p'].diff() != 0).sum()

test['s'] = test['p'] * test['r']
ptc = 0.00012 / test['close']
test['s_'] = np.where(test['p'] != 0, test['s'] - ptc, test['s'])
test[['r', 's', 's_']].sum().apply(np.exp)
test[['r', 's', 's_']].cumsum().apply(np.exp).plot(figsize=(10, 6))

plt.show()
