import csv 
import numpy as np
from numpy.linalg import pinv
import random
import math
import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
with open('best.pickle', 'rb') as f:
    logreg = pickle.load(f)
test_X = pd.read_csv(sys.argv[5])
train_X = pd.read_csv(sys.argv[3])

x = train_X[["age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]]
testing = test_X[["age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]]
# Overfitting
x_frames = [train_X, x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9, x ** 10, 
	x ** 11, x ** 12, x ** 13, x ** 14, x ** 15, x ** 16, x ** 17, x ** 18, x ** 19, x ** 20,
	x ** 21, x ** 22, x ** 23, x ** 24, x ** 25, x ** 26, x ** 27, x ** 28, x ** 29, x ** 30, 
	x ** 31, x ** 32, x ** 33, x ** 34, x ** 35, x ** 36, x ** 37, x ** 38, x ** 39, x ** 40, 
	x ** 41, x ** 42, x ** 43, x ** 44, x ** 45, x ** 46, x ** 47, x ** 48, x ** 49, x ** 50,
	x ** 51, x ** 52, x ** 53, x ** 54, x ** 55,
        np.sin(x), np.cos(x), np.tan(x), np.arctan(x)]
testing_frames = [test_X, testing, testing ** 2, testing ** 3, testing ** 4, testing ** 5, testing ** 6, 
	testing ** 7, testing ** 8, testing ** 9, testing ** 10, testing ** 11, testing ** 12, testing ** 13, testing ** 14, testing ** 15, testing ** 16, 
	testing ** 17, testing ** 18, testing ** 19, testing ** 20, testing ** 21, testing ** 22, testing ** 23, testing ** 24, testing ** 25, testing ** 26, 
	testing ** 27, testing ** 28, testing ** 29, testing ** 30, 
	testing ** 31, testing ** 32, testing ** 33, testing ** 34, testing ** 35, testing ** 36, testing ** 37, testing ** 38, testing ** 39, testing ** 40, 
	testing ** 41, testing ** 42, testing ** 43, testing ** 44, testing ** 45, testing ** 46, testing ** 47, testing ** 48, testing ** 49, testing ** 50,
	testing ** 51, testing ** 52, testing ** 53, testing ** 54, testing ** 55,
	np.sin(testing), np.cos(testing), np.tan(testing), np.arctan(testing)]
testing = pd.concat(testing_frames, axis = 1)
x = pd.concat(x_frames, axis = 1)
scaler = StandardScaler()
scaler.fit(x)
testing = scaler.transform(testing)
y_pred = logreg.predict(testing)

ans = []
yo = y_pred
text = open(sys.argv[6], "w+")
for i in range(len(yo)):
	ans.append([str(i + 1)])
	# a = np.dot(w,yo[i])
	ans[i].append(int(yo[i]))
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
