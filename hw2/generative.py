import csv 
import numpy as np
from numpy.linalg import pinv
import random
import math
import sys
para = np.load("generative.npy")
pinv_sigma = []
def ln_Gaussian(mu_1, mu_2, sigma, test_x):
	global pinv_sigma
	tmp2 = (test_x - mu_2).reshape((-1, 1))
	tmp1 = (test_x - mu_1).reshape((-1, 1))
	return -(1 / 2) * (np.dot(np.dot(tmp2.transpose(), pinv_sigma), tmp2) - np.dot(np.dot(tmp1.transpose(), pinv_sigma), tmp1))
pinv_sigma, mu_1, mu_2, sigma, N_class1, N_class2 = para[0], para[1], para[2], para[3], para[4], para[5]
yo = []
testing, initialized = [], False
test_X = open(sys.argv[5], 'r', encoding='big5') 
row_test_X = csv.reader(test_X , delimiter=",")
count = 0
for r in row_test_X:
	if(not initialized):
		initialized = True
		continue
	testing.append([float(i) for i in r] )
testing = np.array(testing)
for i in testing:
	temp = ln_Gaussian(mu_1, mu_2, sigma, i)
	# print(temp)
	p = 1 / (1 + math.exp(temp) * N_class2 / N_class1)
	print("number:" + str(count), p)
	count += 1
	if(p > 0.5):
		yo.append(1)
	else: 
		yo.append(0)
#print(ans)
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
