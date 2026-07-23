# import kagglehub

# path = kagglehub.dataset_download("uciml/mushroom-classification")

# print(path) 

import pandas as pd 

df = pd.read_csv("mushrooms.csv")

print(df.head())

print("this is the description of the  dataset")

print(df.describe())







