# Importing modules
from regression import Regression
import csv

# Testing
import pandas as pd
import statsmodels.formula.api as sm


path = "/Users/martin/Downloads/Econometrics/Housing.csv"
data = []
with open(path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append([i for i in row[:4]])
        # data.append([i for i in row)

model = Regression(data, header=True)
print(model.ols())

df = pd.read_csv(path)
df_skim = df[['price', 'area', 'bedrooms', 'bathrooms']]
result = sm.ols(formula="price ~ area + bedrooms + bathrooms - 1", data=df).fit()
print(result.params)