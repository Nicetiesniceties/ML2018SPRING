import csv
import numpy as np
from numpy.linalg import pinv
import random
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
	testing_x = np.array(testing_x)
	testing_x = np.concatenate((testing_x, testing_x ** 3), axis=1)
	testing_x = np.concatenate((np.ones((testing_x.shape[0], 1), dtype = float), testing_x), axis = 1)
	# print(np.dot(testing_x, theta))
#2}}}
#{{{2 reading npy
# read model
	theta = np.load('train_best.npy')
	w = np.load('train_best_sample.npy')
#2}}}
#{{{2 Save to csv
	test_x = []
	n_row = 0
	text = open(sys.argv[1] ,"r")
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


	ans = []
	sample = []
	for i in range(len(test_x)):
		sample.append(["id_"+str(i)])
		a = np.dot(w,test_x[i])
		sample[i].append(a)
	for i in range(len(testing_x)):
		ans.append(["id_"+str(i)])
		a = np.dot(theta, testing_x[i])
		ans[i].append(a)
	for i in range(len(ans)):
		if(ans[i][1] < 1 or ans[i][1] > 120):
			ans[i][1] = sample[i][1]
			# print("id", i, sample[i][1])
	filename = sys.argv[2]
	text = open(filename, "w+")
	s = csv.writer(text, delimiter=',',lineterminator='\n')
	s.writerow(["id","value"])
	for i in range(len(ans)):
		s.writerow(ans[i]) 
	text.close()
	print("Sucessfully running hw1_best.py %s %s" % (sys.argv[1], sys.argv[2]))
#2}}}
#}}}
