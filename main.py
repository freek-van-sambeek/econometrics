# Importing modules
from regression import Regression
import hypothesis_testing
import csv

# Testing
import pandas as pd
import statsmodels.formula.api as sm
import timeit


# test_statistic = 0.920
# t = hypothesis_testing.t(degrees_freedom=5)
# print(t.test(test_statistic))
#
# test_statistic = 1.96
# Z = hypothesis_testing.Z()
# print(Z.test(test_statistic))

# test_statistic = 9.24
# Chi_squared = hypothesis_testing.ChiSquared(degrees_freedom=5)
# print(Chi_squared.test(test_statistic))

# test_statistic = 3.63
# F = hypothesis_testing.F(degrees_freedom1=2, degrees_freedom2=16)
# print(F.test(test_statistic))

# path = "/Users/martin/Downloads/Econometrics/Housing.csv"
# data = []
# with open(path, 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         data.append([i for i in row[:4]])
#         # data.append([i for i in row)
#
# start = timeit.default_timer()
# model = Regression(data, header=True)
# print(model.ols().data)
# stop = timeit.default_timer()
# print("Own ran in: ", stop - start)
#
# start = timeit.default_timer()
# df = pd.read_csv(path)
# df_skim = df[['price', 'area', 'bedrooms', 'bathrooms']]
# result = sm.ols(formula="price ~ area + bedrooms + bathrooms - 1", data=df).fit()
# print(result.params)
# stop = timeit.default_timer()
# print("SM ran in: ", stop - start)