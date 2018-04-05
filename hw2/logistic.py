import csv 
import numpy as np
from numpy.linalg import pinv
import random
import math
import sys
w = np.load("logistic.npy")
testing, initialized = [], False
test_X = open(sys.argv[5], 'r', encoding='big5') 
row_test_X = csv.reader(test_X , delimiter=",")
for r in row_test_X:
	if(not initialized):
		initialized = True
		continue
	testing.append([float(i) for i in r] )
testing = np.array(testing)
testing = np.concatenate((testing[:, 0:10], testing[:, 11:]), axis = 1)
testing = np.concatenate((np.ones((testing.shape[0], 1)), testing), axis = 1)
yo, count = [], 0
p = np.dot(testing, w)
for i in p:
	print("number:" + str(count), i)
	count += 1
	if(i > 0):
		yo.append(1)
	else: 
		yo.append(0)
print(yo)
# print ('epoch: %d | w_length: %f |Cost: %f  ' % (i, np.sum(w ** 2), temp_n / x.shape[0]))



ans = []
text = open(sys.argv[6], "w+")
for i in range(len(yo)):
	ans.append([str(i + 1)])
	# a = np.dot(w,yo[i])
	ans[i].append(int(yo[i]))
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
