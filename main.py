import pandas as pd
import csv
import tensorflow as tf
import numpy as np


#should have a main method here that calls preprocess, for now will just include pos or neg function here to add labels
def pos_or_neg(a):
    if a > 0:
        to_return = 1
    elif a < 0:
        to_return = -1
    else:
        to_return = 0
    return to_return
    

df= pd.read_csv('sp500.csv')
dk = pd.read_csv('aapl.csv')
real_data = df[['Adj Close', 'Volume']]
df['dif in close'] = df['Adj Close'].diff()
df['labels'] = df.apply(lambda x: pos_or_neg(x['dif in close']), axis=1)
real_data_with_labels = df[['Adj Close', 'Volume', 'labels']]


nice = dk[['Adj Close', 'Volume']]
dk['Prev_14_Close'] = dk['Close'].shift(14)
dk['Prev_14_Volume'] = dk['Volume'].shift(14)

# dk.head(20)

# # Drop the rows with missing values (since the first 14 rows will have NaN values)
# previous_close_points = dk['Prev_14_Close'].dropna().tolist()
# previous_volume_points = dk['Prev_14_Volume'].dropna().tolist()
# nparray = []
# realarray = []
# # print(nice)
# # Print the resulting DataFrame
# for i in range(len(nice)):
#     if i > 13:
    
#         nparray = np.append(nparray, nice[i - 14: i])
        
# # print(nparray.shape)
# reshaped_array = np.reshape(nparray, (-1, 14, 2))

# # Print the reshaped array
# # print(reshaped_array.shape)
# # print(reshaped_array)

# for i in range(len(real_data)):
#     if i > 13:
#         realarray = np.append(realarray, real_data[i - 14: i])
        
# real_reshape = np.reshape(realarray, (-1, 14, 2))
# # print(real_reshape.shape)
# # nice = tf.keras.layers.Reshape((-1, 14, 2))(nice)
# # volume = df['Volume']
# # adjusted_close = df['Adj Close']
# # print(volume)
# # print(adjusted_close)
# # print(nice.shape)
# # print(volume)
# # print(adjusted_close)
# pd.read_csv('aapl.csv').head()





# apple_test = pd.read_csv('aapl.csv')
# apple_test['dif in close'] = apple_test['Close'].diff()
# apple_test['labels'] = apple_test.apply(lambda x: pos_or_neg(x['dif in close']), axis=1)
# # print(apple_test.iloc[5])
# # print(apple_test.iloc[2])
# # print(apple_test.iloc[5].shift(3))
# # apple_test['Close'].shift(1-14).head(20)
# # print(apple_test['Close'].rolling(14))
# # apple_test['inputs'] = apple_test['Close'].rolling(14).apply(lambda x: list(np.array(x.shift(1-14))))
# # apple_test.head(20)


