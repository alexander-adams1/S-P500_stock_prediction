from __future__ import absolute_import
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
from preprocess import read_csv
from preprocess import get_labels
from preprocess import get_inputs
from preprocess import pos_or_neg
import random
from sklearn.model_selection import train_test_split
import statistics






def main():
    sp500_dataframe = read_csv('sp500.csv')
    aapl_dataframe = read_csv('aapl.csv')

    sp500_labels = get_labels(sp500_dataframe)
    # aapl_with_labels = get_labels(aapl_dataframe)

    close_inputs, vol_inputs = get_inputs(sp500_dataframe)
    # apple_inputs = get_inputs(aapl_dataframe)

    # inputs = full_inputs
    labels = sp500_labels



    # # print(sp500_with_labels)
    # data = list(zip(inputs, labels))
    # # print(data)
    # data = tf.random.shuffle(data)
    
    
    # shuffled_inputs, shuffled_labels = zip(*data)
    # # print(np.shape(shuffled_inputs))
    # # print(np.shape(shuffled_labels))
    # n  = int(len(shuffled_inputs) * 0.8)
    
    # train_inputs = shuffled_inputs[: n]
    # test_inputs = shuffled_inputs[n: ]
    # train_labels = shuffled_labels[: n]
    # test_labels = shuffled_labels[n: ]

    # print(tf.shape(train_inputs))

    # print(tf.shape(inputs))
    # print(tf.shape(labels))

    train_close_inputs, test_close_inputs, train_close_labels, test_close_labels = train_test_split(close_inputs, labels, test_size=.2, shuffle=True)
    train_vol_inputs, test_vol_inputs, train_vol_labels, test_vol_labels = train_test_split(vol_inputs, labels, test_size=.2, shuffle=True)


    train_close_inputs = tf.convert_to_tensor(np.reshape(train_close_inputs, (-1, 14, 1)))
    test_close_inputs = tf.convert_to_tensor(np.reshape(test_close_inputs, (-1, 14, 1)))
    train_close_labels = tf.convert_to_tensor(train_close_labels)
    test_close_labels = tf.convert_to_tensor(test_close_labels)

    train_vol_inputs = tf.convert_to_tensor(np.reshape(train_vol_inputs, (-1, 14, 1)))
    test_vol_inputs = tf.convert_to_tensor(np.reshape(test_vol_inputs, (-1, 14, 1)))
    train_vol_labels = tf.convert_to_tensor(train_vol_labels)
    test_vol_labels = tf.convert_to_tensor(test_vol_labels)

    model_close = Model()
    num_batches = len(train_close_inputs) // model_close.batch_size
    test_batches = len(test_close_inputs) // model_close.batch_size
    # print(num_batches)
    # print(test_batches)
    # print(tf.shape(train_close_inputs))
    for i in range(10):
        print(train(model_close, train_close_inputs, train_close_labels, num_batches))
        print("closing price: finished epoch")
        print(i)
        print(test(model_close, test_close_inputs, test_close_labels, test_batches)) 



    model_vol = Model()
    for i in range(10):
        print(train(model_vol, train_vol_inputs, train_vol_labels, num_batches))
        print("volume: finished epoch")
        print(i)
        print(test(model_vol, test_vol_inputs, test_vol_labels, test_batches))
    
    # print(test(model, test_inputs, test_labels, test_batches))  
    


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 32
        self.loss_list = []
        self.flatten = tf.keras.layers.Flatten()
        self.conv_weights = tf.cast(tf.random.normal([3, 1, 4]), dtype='float64')
        # self.conv_layer = tf.keras.layers.Conv1D(filters=4, kernel_size=3, activation='relu', kernel_initializer=self.conv_weights)
        self.dense1 = tf.keras.layers.Dense(64, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='linear')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.003)

    def call(self, inputs):

        
        # print('begin check')
        # print(self.conv_weights)
        # print('end check')
        conv_layer = tf.nn.conv1d(inputs, filters = self.conv_weights, stride=1, padding='SAME')
        flatten = self.flatten(conv_layer)
        dense = self.dense1(flatten)
        dense2 = self.dense2(dense)

            
        # for i in range(np.shape(inputs)[0]):
        # conv_layer = self.conv_layer(inputs)
        # flatten = self.flatten(inputs)
        # dense = self.dense(flatten)
        return dense2


    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        # print()
        return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))


    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        

def train(model, train_inputs, train_labels, num_batches):
    total_loss = 0
    # # print(tf.shape(train_labels))

    # for batch in range(num_batches):
    #     batch_start = batch * model.batch_size
    #     batch_end = (batch + 1) * model.batch_size
    #     batch_labels = train_labels[batch_start:batch_end]
    #     batch_inputs = train_inputs[batch_start:batch_end]
    #     # print(batch_start)
    #     # print(batch_end)
    #     # print(batch_inputs)
    #     # print(tf.shape(batch_labels))
    #     # print(np.shape(batch_inputs))
    #     for i in range(np.shape(batch_inputs)[0]):
    #         # print(np.shape(inputs[i]))
    #         output = tf.transpose(batch_inputs[i])
    #         # print(output[0])
    #         output = tf.reshape(output[0], (1, 14, 1))
    #         output = tf.cast(output, dtype = 'float64')
    #         with tf.GradientTape() as tape:
    #             logits = model.call(output)
    #             loss = model.loss(logits, batch_labels)
    #             # print(batch_labels[i])
    #             # print(batch_labels.shape())
    #             # print(loss)
    #             # print("logits")
    #             # print(logits)
    #             # print("logits")
    #             # print("labels")
    #             # print(batch_labels[i])
    #             # print("labels")
    #         grads = tape.gradient(loss, model.trainable_variables)
    #         model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #         total_loss += loss
    # print("lans")
    # print(tf.shape(train_labels)[0])
    # print("fini")
    loss_list = []
    for b, b1 in enumerate(range(model.batch_size, tf.shape(train_labels)[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        with tf.GradientTape() as tape:
            pred = model.call(train_inputs[b0:b1])
            loss = model.loss(pred, train_labels[b0:b1])    
        gradients = tape.gradient(loss, model.trainable_variables)
        # print(gradients)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_list.append(loss)
    return statistics.fmean(loss_list)


    
def test(model, test_inputs, test_labels, test_batches):
    accuracy = []
    for b, b1 in enumerate(range(model.batch_size, tf.shape(test_labels)[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        logits = model.call(test_inputs[b0:b1])
        accuracy.append(model.accuracy(logits, test_labels[b0:b1]))
    accuracy = tf.Variable(accuracy)
    accuracy = tf.math.reduce_mean(accuracy)
    return accuracy




    # accuracy = []
    # for batch in range(test_batches):
    #     batch_start = batch * model.batch_size
    #     batch_end = (batch + 1) * model.batch_size
    #     batch_labels = test_labels[batch_start:batch_end]
    #     batch_inputs = test_inputs[batch_start:batch_end]
    #     # with tf.GradientTape() as tape:
    #     logits = model.call(batch_inputs)
    #     # print(model.accuracy(logits, batch_labels))
    #     accuracy.append(model.accuracy(logits, batch_labels))
    #     # grads = tape.gradient(loss, model.trainable_variables)
    #     # model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # accuracy = tf.math.reduce_mean(accuracy)
    # return accuracy


    

# df= pd.read_csv('sp500.csv')
# dk = pd.read_csv('aapl.csv')
# real_data = df[['Adj Close', 'Volume']]
# df['dif in close'] = df['Adj Close'].diff()
# df['labels'] = df.apply(lambda x: pos_or_neg(x['dif in close']), axis=1)
# real_data_with_labels = df[['Adj Close', 'Volume', 'labels']]

# # print(real_data_with_labels)

# nice = dk[['Adj Close', 'Volume']]
# dk['Prev_14_Close'] = dk['Close'].shift(14)
# dk['Prev_14_Volume'] = dk['Volume'].shift(14)

# dk.head(20)

# Drop the rows with missing values (since the first 14 rows will have NaN values)
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
# print('test reshaped array shape: ', reshaped_array.shape)
# # print(reshaped_array)

# for i in range(len(real_data)):
#     if i > 13:
#         realarray = np.append(realarray, real_data[i - 14: i])
        
# real_reshape = np.reshape(realarray, (-1, 14, 2))
# print(real_reshape.shape)
# print(real_reshape)
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


if __name__ == '__main__':
    main()