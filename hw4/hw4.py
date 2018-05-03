import csv
import sys
ans, text = [], open(sys.argv[3], "w+")
for i in range(1980000):
	ans.append([str(i)])
	ans[i].append(int(0))
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["ID","Ans"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
