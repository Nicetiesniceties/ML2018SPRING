import csv
import numpy as np
from numpy.linalg import pinv
import math
import sys
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
	# parsing data #
	index = 240
	for row in range(len(x)):
		for i in range(len(x[row])):
			if x[row][i] < 1 or x[row][i] > 120:
				x = np.delete(x, (row), axis = 0)
				y = np.delete(y, (row), axis = 0)
				index, i = index - 1, 0
		if row == index - 1:
			break
	# ------------ #
	x = np.concatenate((x, x ** 2), axis=1)
	# x_temp = np.concatenate((x_temp, x ** 3), axis=1)
	# x_temp = np.concatenate((x_temp, x ** 4), axis=1)
	# x = np.concatenate((x_temp, x ** 5), axis=1)
	x = np.concatenate((np.ones((x.shape[0], 1), dtype = float), x), axis = 1)
	theta = np.zeros(len(x[0]))
	theta = theta.transpose()
#2}}}
#{{{2 Closed Form Solution
	# argmin(theta) = (x^T x)^(-1) x^T y
	# theta = np.matmul(np.matmul(pinv(np.matmul(x.transpose(), x)), x.transpose()), y)
	x_t = x.transpose()
	sum_gra = np.zeros(len(x[0]))
	l_rate = 0.005
	reg = 1

	loss = np.dot(x, theta) - y
	#print(loss[0])
	cost = np.sum(loss ** 2) / len(x)
	cost_sqrt  = math.sqrt(cost)
	print ('Cost: %f  ' % (cost_sqrt))
#2}}}
#{{{2 Gradient descent
	for i in range(100000):
		loss = np.dot(x, theta) - y
		cost = np.sum(loss ** 2) / len(x)
		cost_sqrt = math.sqrt(cost)
		gra = np.matmul(x_t, loss) / cost_sqrt + 2 * reg * theta
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
	for i in range(len(testing_x)):
		for j in range(len(testing_x[i])):
			if testing_x[i][j] < 0 or testing_x[i][j] > 120:
				if j == 0:
					testing_x[i][j] = testing_x[i][j + 1]
				elif j == len(testing_x[i]) - 1:
					testing_x[i][j] = testing_x[i][j - 1]
				else:
					testing_x[i][j] = (testing_x[i][j + 1] + testing_x[i][j - 1]) / 2
	testing_x = np.array(testing_x)
	testing_x = np.concatenate((testing_x, testing_x ** 2), axis=1)
	# testing_x_temp = np.concatenate((testing_x_temp, testing_x ** 3), axis=1)
	# testing_x_temp = np.concatenate((testing_x_temp, testing_x ** 4), axis=1)
	# testing_x = np.concatenate((testing_x_temp, testing_x ** 5), axis=1)
	testing_x = np.concatenate((np.ones((testing_x.shape[0], 1), dtype = float), testing_x), axis = 1)
	print(np.dot(testing_x, theta))
#2}}}
#{{{2 save npy
# save model
	np.save('train.npy',theta)
# read model
	theta = np.load('train.npy')
#}}}
#{{{2 Save to csv
	ans = []
	for i in range(len(testing_x)):
		ans.append(["id_"+str(i)])
		a = np.dot(theta, testing_x[i])
		ans[i].append(a)
	for i in range(len(ans)):
		if(ans[i][1] < 1 or ans[i][1] > 120):
			if(i == 0):
				ans[i][1] = ans[i + 1][1]
			elif(i == len(ans) - 1):
				ans[i][1] = ans[i - 1][1]
			else:
				ans[i][1] = (ans[i - 1][1] + ans[i + 1][1]) / 2
	filename = "ans.csv"
	text = open(filename, "w+")
	s = csv.writer(text, delimiter=',',lineterminator='\n')
	s.writerow(["id","value"])
	for i in range(len(ans)):
		s.writerow(ans[i]) 
	text.close()
#2}}}
#}}}
