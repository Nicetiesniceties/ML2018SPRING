import csv
import numpy as np
from numpy.linalg import pinv
import random
import math
# sample{{{
import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# add square term
x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 1
repeat = 10000

# use close form to check whether ur gradient descent is good
# however, this cannot be used in hw1.sh 
# w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)
print(w)
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

##
#hypo = np.dot(x,w)
#loss = hypo - y
#cost = np.sum(loss**2) / len(x)
#cost_a  = math.sqrt(cost)
#gra = np.dot(x_t,loss)
#s_gra += gra**2
#ada = np.sqrt(s_gra)
#print ('Cost: %f  ' % ( cost_a))
##

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

test_x = []
n_row = 0
text = open('test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

sample = []
for i in range(len(test_x)):
    sample.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    sample[i].append(a)
text.close()
#{{{2 saving model
# save model
np.save('train_best_sample.npy',w)
# read model
w = np.load('train_best_sample.npy')
#2}}}
#}}}
# read_train_csv(){{{
def read_train_csv(pathname, raw_data, range_of_cols):
	with open(pathname, 'r', encoding='big5') as csvfile:
		reader = csv.reader(csvfile)
		num = 0
		initialized = False
		for row in reader:
			if initialized == False:
				initialized = True
				continue
			for idx in range_of_cols:
				if row[idx] == "NR":
					raw_data[num % 18].append(float(0))
				else:
					raw_data[num % 18].append(float(row[idx]))
			num += 1
#}}}
# read_test_csv(){{{
def read_test_csv(pathname, testing_data, range_of_cols):
	with open(pathname, 'r', encoding='big5') as csvfile:
		reader = csv.reader(csvfile)
		num = 0
		for row in reader:
			for idx in range_of_cols:
				if row[idx] == "NR":
					testing_data[num % 18].append(float(0))
				else:
					testing_data[num % 18].append(float(row[idx]))
			num += 1
#}}}
# manipulate_raw_data(){{{
def manipulate_training_data(raw_data, x, y):
	i = 0
	for count in range(240):
		x.append([])
		for hour in range(8, 9):
			for feature in range(8, 10):
				x[i].append(raw_data[feature][count * 24 + hour])
		y.append(raw_data[9][count * 24 + 9])
		i += 1
#}}}
# manipulate_testing_data(){{{
def manipulate_testing_data(testing_data, testing_x):
	i = 0
	for count in range(260):
		testing_x.append([])
		for hour in range(8, 9):
			for feature in range(8, 10):
				testing_x[i].append(float(testing_data[feature][count * 9 + hour]))
		i += 1
#}}}
# main(){{{
if __name__ == '__main__':
	raw_data, testing_data, testing_x, x, y = [], [], [], [], []
#{{{2 Handling training data
	for i in range(18):
		raw_data.append([])
	read_train_csv("train.csv", raw_data, range(3, 27))
	manipulate_training_data(raw_data, x, y)
	x, y = np.array(x), np.array(y)
	y = y.transpose()
	x = np.concatenate((x, x ** 3), axis=1)
	x = np.concatenate((np.ones((x.shape[0], 1), dtype = float), x), axis = 1)
	theta = np.zeros(len(x[0]))
	theta = theta.transpose()
#2}}}
#{{{2 Closed Form Solution
	# argmin(theta) = (x^T x)^(-1) x^T y
	theta = np.matmul(np.matmul(pinv(np.matmul(x.transpose(), x)), x.transpose()), y)
	x_t = x.transpose()
	sum_gra = np.zeros(len(x[0]))
	l_rate = 0.00005
#2}}}
#{{{2 Gradient descent
	for i in range(0):
		loss = np.dot(x, theta) - y
		cost = np.sum(loss ** 2) / len(x)
		cost_sqrt = math.sqrt(cost)
		gra = np.matmul(x_t, loss) / cost_sqrt
		sum_gra += gra ** 2
		adagrad = np.sqrt(sum_gra)
		# print(theta)
		# print("theta: %f, l_rate: %f, gra: %f, adagrad: %f" %(theta[22], l_rate, gra[22], adagrad[22]))
		theta = theta - l_rate * gra / adagrad
		print ('iteration: %d | Cost: %f  ' % ( i, cost_sqrt))
#2}}}
#{{{2Handling testing data
	for i in range(18):
		testing_data.append([])
	read_test_csv("test.csv", testing_data, range(2, 11))
	manipulate_testing_data(testing_data, testing_x)
	testing_x = np.array(testing_x)
	testing_x = np.concatenate((testing_x, testing_x ** 3), axis=1)
	testing_x = np.concatenate((np.ones((testing_x.shape[0], 1), dtype = float), testing_x), axis = 1)
	print(np.dot(testing_x, theta))
#2}}}
#{{{2 saving npy
# save model
	np.save('train_best.npy',theta)
# read model
	theta = np.load('train_best.npy')
#2}}}
#{{{2 Save to csv
	ans = []
	for i in range(len(testing_x)):
		ans.append(["id_"+str(i)])
		a = np.dot(theta, testing_x[i])
		ans[i].append(a)
	for i in range(len(ans)):
		if(ans[i][1] < 1 or ans[i][1] > 120):
			ans[i][1] = sample[i][1]
			print("id", i, sample[i][1])
	filename = "ans.csv"
	text = open(filename, "w+")
	s = csv.writer(text, delimiter=',',lineterminator='\n')
	s.writerow(["id","value"])
	for i in range(len(ans)):
		s.writerow(ans[i]) 
	text.close()
#2}}}
#}}}
