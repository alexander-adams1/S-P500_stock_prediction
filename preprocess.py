import pandas as pd
import csv
import tensorflow as tf
import numpy as np
import os



def read_csv(filepath):
    dataframe = pd.read_csv(filepath)
    return dataframe

def get_labels(dataframe):
    dataframe['dif in close'] = dataframe['Adj Close'].diff()
    dataframe['labels'] = dataframe.apply(lambda x: pos_or_neg(x['dif in close']), axis=1)
    return dataframe

def get_inputs(dataframe):
    #return inputs which should be converted from the passed in dataframe to a numpy array
    #also convert the labels in the dataframe as part of the array
    #returns the array we need
    inputs = None
    labels = None
    return inputs, labels


def pos_or_neg(a):
    if a > 0:
        to_return = 1
    elif a < 0:
        to_return = -1
    else:
        to_return = 0
    return to_return


