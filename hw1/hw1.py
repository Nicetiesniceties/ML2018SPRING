import csv
import numpy as np
from numpy.linalg import pinv
import math
import sys
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
#{{{2Handling testing data
	for i in range(18):
		testing_data.append([])
	read_test_csv(sys.argv[1], testing_data, range(2, 11))
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
#2}}}
#{{{2 save npy
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
	filename = sys.argv[2]
	text = open(filename, "w+")
	s = csv.writer(text, delimiter=',',lineterminator='\n')
	s.writerow(["id","value"])
	for i in range(len(ans)):
		s.writerow(ans[i]) 
	text.close()
	print("Sucessfully running hw1.py %s %s" % (sys.argv[1], sys.argv[2]))
#2}}}
#}}}
