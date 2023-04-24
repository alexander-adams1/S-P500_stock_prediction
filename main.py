import pandas as pd
import csv


df= pd.read_csv('sp500.csv')
volume = df['Volume']
adjusted_close = df['Adj Close']
print(volume)
print(adjusted_close)

