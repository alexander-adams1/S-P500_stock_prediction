import pandas as pd
import csv
import tensorflow as tf
import numpy as np


df= pd.read_csv('sp500.csv')
dk = pd.read_csv('aapl.csv')
nice = dk[['Close', 'Volume']]
nice = tf.keras.layers.Reshape((-1, 14, 2))(nice)
# volume = df['Volume']
# adjusted_close = df['Adj Close']
# print(volume)
# print(adjusted_close)
print(nice.shape)
# print(volume)
# print(adjusted_close)
pd.read_csv('aapl.csv').head()


def pos_or_neg(a):
    if a > 0:
        to_return = 1
    elif a < 0:
        to_return = -1
    else:
        to_return = 0
    return to_return


apple_test = pd.read_csv('aapl.csv')
apple_test['dif in close'] = apple_test['Close'].diff()
apple_test['labels'] = apple_test.apply(lambda x: pos_or_neg(x['dif in close']), axis=1)
print(apple_test.iloc[5])
print(apple_test.iloc[2])
print(apple_test.iloc[5].shift(3))
# apple_test['Close'].shift(1-14).head(20)
# print(apple_test['Close'].rolling(14))
# apple_test['inputs'] = apple_test['Close'].rolling(14).apply(lambda x: list(np.array(x.shift(1-14))))
# apple_test.head(20)


