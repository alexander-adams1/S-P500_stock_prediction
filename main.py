from __future__ import absolute_import
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
from preprocess import get_inputs_2, read_csv
from preprocess import get_labels
from preprocess import get_inputs
from preprocess import pos_or_neg
import random
from sklearn.model_selection import train_test_split
import statistics
from matplotlib import pyplot as plt







def main():
    sp500_dataframe = read_csv('sp500.csv')
    aapl_dataframe = read_csv('aapl.csv')

    sp500_labels = get_labels(sp500_dataframe)
    # aapl_with_labels = get_labels(aapl_dataframe)

    close_inputs, vol_inputs = get_inputs(sp500_dataframe)
    dual_inputs = get_inputs_2(sp500_dataframe)
    # apple_inputs = get_inputs(aapl_dataframe)
    print(np.shape(close_inputs))
    print(np.shape(vol_inputs))
    # inputs = full_inputs
    labels = sp500_labels
    
    # dual_inputs = tf.concat([close_inputs, vol_inputs], axis = -1)
    dual_inputs = np.reshape(dual_inputs, (-1, 14, 2))

    # dual_inputs = tf.reshape(dual_inputs, (-1, 2, 14))
    # print(np.shape(dual_inputs))
    # print(dual_inputs)
    # # print(sp500_with_labels)
    # print(data)
    
    n = len(labels)
    indices = list(range(n))

# shuffle the list of numbers
    random.shuffle(indices)
   
    shuffled_inputs = [dual_inputs[i] for i in indices]
    shuffled_labels = [labels[i] for i in indices]
    # # print(np.shape(shuffled_inputs))
    # # print(np.shape(shuffled_labels))
    n  = int(len(shuffled_inputs) * 0.8)
    
    train_inputs = shuffled_inputs[: n]
    test_inputs = shuffled_inputs[n: ]
    train_labels = shuffled_labels[: n]
    test_labels = shuffled_labels[n: ]

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
    
   

    
    
    # print(num_batches)
    # print(test_batches)
    # print(tf.shape(train_close_inputs))
    # train_inputs = tf.concat([train_close_inputs, train_vol_inputs], axis = 1)
    # train_labels = tf.concat([train_close_labels, train_vol_labels], axis = 1)
    # train_inputs = tf.reshape(train_inputs, (-1, 2, 14))
    # train_labels = tf.reshape(train_labels, (-1, 2, 14))
    
    # test_inputs = tf.concat([test_close_inputs, test_vol_inputs], axis = 1)
    # test_labels = tf.concat([test_close_labels, test_vol_labels], axis = 1)
    # test_inputs = tf.reshape(test_inputs, (-1, 2, 14))
    # test_labels = tf.reshape(test_labels, (-1, 2, 14))
    
    
    dual_model_loss = []
    close_model_loss = []
    vol_model_loss = []
    
    dual_model = Model()
    for i in range(60):
        to_print, losses = train_2(dual_model, train_inputs, train_labels)
        print(to_print)
        print("dual model: finished epoch")
        print(i)
        dual_model_loss.append(losses)
    visualize_loss(dual_model_loss)
    print(test_2(dual_model, test_inputs, test_labels)) 
        
    model_close = Model()  
    num_batches = len(train_close_inputs) // model_close.batch_size
    test_batches = len(test_close_inputs) // model_close.batch_size  
    for i in range(30):
        to_print1, losses1 = train(model_close, train_close_inputs, train_close_labels, num_batches, is_vol=False)
        print(to_print1)
        print("closing price: finished epoch")
        print(i)
        close_model_loss.append(losses1)
    visualize_loss(close_model_loss)
    print(test(model_close, test_close_inputs, test_close_labels, test_batches)) 



    model_vol = Model()
    for i in range(30):
        to_print2, losses2 = train(model_vol, train_vol_inputs, train_vol_labels, num_batches, is_vol=True)
        print(to_print2)
        print("volume: finished epoch")
        print(i)
        vol_model_loss.append(losses2) 
    visualize_loss(vol_model_loss)
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=.0005)
        self.optimizer_dual = tf.keras.optimizers.Adam(learning_rate=.000065)
        # self.optimizer_volume = tf.keras.optimizers.Adam(learning_rate=.00008)
        
        # self.seq = tf.keras.Sequential([
        #     # tf.keras.layers.Input(shape=(14, 2)),
        #     tf.keras.layers.Conv1D(filters=4, kernel_size=3, activation='relu'),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(units=10, activation='relu'),
        #     tf.keras.layers.Dense(units=2, activation='linear')
        # ])
        
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
    
    def call_2(self, inputs):
        inputs = np.array(inputs)
        first_half = inputs[:, :, :1]
        # second_half = inputs[: , :2, :]
        second_half = inputs[:, :, 1:]
        conv_value_layer = tf.nn.conv1d(first_half, filters = self.conv_weights, stride=1, padding='SAME')
        conv_volume_layer = tf.nn.conv1d(second_half, filters = self.conv_weights, stride=1, padding='SAME')
        flatten_first = self.flatten(conv_value_layer)
        flatten_second = self.flatten(conv_volume_layer)
        final_conv_layer = tf.concat([flatten_first, flatten_second], axis = -1)
        dense = self.dense1(final_conv_layer)
        dense2 = self.dense2(dense)
        
        
        # outputs = self.seq(inputs)
        
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

        

def train(model, train_inputs, train_labels, num_batches, is_vol):
    total_loss = 0
    indices = tf.range(len(train_inputs))
    indices = tf.random.shuffle(indices)
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)
    loss_list = []
    for b, b1 in enumerate(range(model.batch_size, tf.shape(train_labels)[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        with tf.GradientTape() as tape:
            pred = model.call(train_inputs[b0:b1]) 
            loss = model.loss(pred, train_labels[b0:b1])    
        gradients = tape.gradient(loss, model.trainable_variables)
        # print(gradients)
        if is_vol:
            model.optimizer2.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_list.append(loss)
    # visualize_loss(loss_list)
    return (statistics.fmean(loss_list), loss_list)

    
def test(model, test_inputs, test_labels, test_batches):
    accuracy = []
    for b, b1 in enumerate(range(model.batch_size, tf.shape(test_labels)[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        logits = model.call(test_inputs[b0:b1])
        accuracy.append(model.accuracy(logits, test_labels[b0:b1]))
    accuracy = tf.Variable(accuracy)
    accuracy = tf.math.reduce_mean(accuracy)
    return accuracy

def train_2(model, train_inputs, train_labels):
    loss_list = []
    indices = tf.range(len(train_inputs))
    indices = tf.random.shuffle(indices)
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)
    for b, b1 in enumerate(range(model.batch_size, tf.shape(train_labels)[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        with tf.GradientTape() as tape:
            pred = model.call_2(train_inputs[b0:b1])
            
            loss = model.loss(pred, train_labels[b0:b1])    
        gradients = tape.gradient(loss, model.trainable_variables)
        # print(gradients)
        model.optimizer_dual.apply_gradients(zip(gradients, model.trainable_variables))
        loss_list.append(loss)
    # visualize_loss(loss_list)
    return (statistics.fmean(loss_list), loss_list)

def test_2(model, test_inputs, test_labels):
    accuracy = []
    for b, b1 in enumerate(range(model.batch_size, tf.shape(test_labels)[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        logits = model.call_2(test_inputs[b0:b1])
        accuracy.append(model.accuracy(logits, test_labels[b0:b1]))
    accuracy = tf.Variable(accuracy)
    accuracy = tf.math.reduce_mean(accuracy)
    return accuracy


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()  



if __name__ == '__main__':
    main()