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




def main():
    sp500_dataframe = read_csv('sp500.csv')
    aapl_dataframe = read_csv('aapl.csv')

    sp500_labels = get_labels(sp500_dataframe)
    # aapl_with_labels = get_labels(aapl_dataframe)

    full_inputs = get_inputs(sp500_dataframe)
    # apple_inputs = get_inputs(aapl_dataframe)

    inputs = full_inputs
    labels = sp500_labels

    # print(sp500_with_labels)
    data = list(zip(inputs, labels))
    random.shuffle(data)
    
    
    shuffled_inputs, shuffled_labels = zip(*data)
    print(np.shape(shuffled_inputs))
    print(np.shape(shuffled_labels))
    n  = int(len(shuffled_inputs) * 0.8)
    
    train_inputs = shuffled_inputs[: n]
    test_inputs = shuffled_inputs[n: ]
    train_labels = shuffled_labels[: n]
    test_labels = shuffled_labels[n: ]
    

    model = Model()
    num_batches = len(train_inputs) // model.batch_size
    for i in range(10):
        train(model, train_inputs, train_labels, num_batches)
        test(model, test_inputs, test_labels, num_batches)   
    

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 32
        self.loss_list = []
        self.flatten = tf.keras.layers.Flatten()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.003)

    def call(self):
        print(nice)
        return 1


    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
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

        

def train(self, model, train_inputs, train_labels, num_batches):
    total_loss = 0

    for batch in range(num_batches):
        batch_start = batch * model.batch_size
        batch_end = (batch + 1) * model.batch_size
        batch_labels = train_labels[batch_start:batch_end]
        batch_inputs = train_inputs[batch_start:batch_end]
        with tf.GradientTape() as tape:
            logits = self.call(batch_inputs)
            loss = model.loss(logits, batch_labels)
            grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        total_loss += loss
    return total_loss
    
def test(self, model, test_inputs, test_labels, num_batches):
    accuracy = []
    for batch in range(num_batches):
        batch_start = batch * model.batch_size
        batch_end = (batch + 1) * model.batch_size
        batch_labels = test_labels[batch_start:batch_end]
        batch_inputs = test_inputs[batch_start:batch_end]
        # with tf.GradientTape() as tape:
        logits = self.call(batch_inputs)
        accuracy.append(model.accuracy(logits, batch_labels))
        # grads = tape.gradient(loss, model.trainable_variables)
        # model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    accuracy = tf.math.reduce_mean(accuracy)
    return accuracy


    

df= pd.read_csv('sp500.csv')
dk = pd.read_csv('aapl.csv')
real_data = df[['Adj Close', 'Volume']]
df['dif in close'] = df['Adj Close'].diff()
df['labels'] = df.apply(lambda x: pos_or_neg(x['dif in close']), axis=1)
real_data_with_labels = df[['Adj Close', 'Volume', 'labels']]

# print(real_data_with_labels)

nice = dk[['Adj Close', 'Volume']]
dk['Prev_14_Close'] = dk['Close'].shift(14)
dk['Prev_14_Volume'] = dk['Volume'].shift(14)

dk.head(20)

# Drop the rows with missing values (since the first 14 rows will have NaN values)
previous_close_points = dk['Prev_14_Close'].dropna().tolist()
previous_volume_points = dk['Prev_14_Volume'].dropna().tolist()
nparray = []
realarray = []
# print(nice)
# Print the resulting DataFrame
for i in range(len(nice)):
    if i > 13:
    
        nparray = np.append(nparray, nice[i - 14: i])
        
# print(nparray.shape)
reshaped_array = np.reshape(nparray, (-1, 14, 2))

# Print the reshaped array
print('test reshaped array shape: ', reshaped_array.shape)
# print(reshaped_array)

for i in range(len(real_data)):
    if i > 13:
        realarray = np.append(realarray, real_data[i - 14: i])
        
real_reshape = np.reshape(realarray, (-1, 14, 2))
print(real_reshape.shape)
print(real_reshape)
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