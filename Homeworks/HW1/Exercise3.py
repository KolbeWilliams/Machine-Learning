#Exercise 3:
import pandas as pd

#All lockers start closed (0)
lockers_dict = {i: 0 for i in range(1, 101)}
lockers = pd.Series(lockers_dict)
#First student opens all lockers (1)
for i in range(101):
    lockers[i] = 1
#Second student closes every second locker
for i in range(0, 101, 2):
    lockers[i] = 0
#All other students
for x in range(3,len(lockers)):
    for i in range(0, 101, x):
         lockers[i] = 0 if lockers[i] == 1 else 1
lockers = lockers.drop(0) #drop the last row that was added in the for loops

for i in range(1, 101):
    if lockers[i] == 1:
        print(f'locker {i} is open')