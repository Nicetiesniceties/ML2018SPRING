# -*- coding: utf-8 -*-
"""democracy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zYA_Yk6zHBY6nkJtC20lnaqg7FWSoksU
"""
#residual learning
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import csv
import pandas as pd
from keras.utils import np_utils
from pandas import to_numeric
import random
import sys
np.random.seed(25)
testing_set = pd.read_csv(sys.argv[1])
# print(testing_set)
test_id = testing_set[["id"]]
raw_test_x = testing_set[["feature"]]
# print(raw_test_x)
# print(test_id)
raw_test_x = np.array(raw_test_x)
temp_test_x = []
for i in raw_test_x:
  temp_test_x.append(i[0].split(" "))
test_x = np.zeros((len(temp_test_x), 48 * 48))
for i in range(len(temp_test_x)):
  for j in range(len(temp_test_x[i])):
    test_x[i][j] = float(temp_test_x[i][j])

# model.save_weights("drive/model/10addmoreconvandNN.h5")

from keras.models import load_model
# model.load_weights("drive/model/10addmoreconvandNN.h5")
test_x = test_x.reshape(len(test_x), 48, 48, 1)
test_x_normalized = test_x / 255
# results = model.predict_classes(test_x_normalized)

# m0 = model.load_weights("3addmoreconv_overfitting.h5")
# m1 = model.load_weights("4goodregconv.h5")
# m2 = model.load_weights("5goodreglessdropoutconv.h5")
# m3 = model.load_weights("6moreNNandConv.h5")
m4 = load_model("medium062.h5")
m6 = load_model("medium064.h5")
m7 = load_model("notmedium062.h5")
p4 = m4.predict(test_x_normalized)
p6 = m6.predict(test_x_normalized)
p7 = m7.predict(test_x_normalized)
# m8 = model.load_weights("11addNN.h5")
p = p4 * 0.62 + p6 * 0.64 + p7 * 0.62
results = np.argmax(p, axis = 1)
print(results)

ans = []
print(results)
text = open(sys.argv[2], "w+")
for i in range(len(results)):
	ans.append([str(i)])
	# a = np.dot(w,yo[i])
	ans[i].append(int(results[i]))
print(ans)
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
   s.writerow(ans[i])

print("Love never felt so good~~~")

ans = []
print(results)
text = open(sys.argv[2], "w+")
for i in range(len(results)):
	ans.append([str(i)])
	# a = np.dot(w,yo[i])
	ans[i].append(int(results[i]))
print(ans)
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
   s.writerow(ans[i])
