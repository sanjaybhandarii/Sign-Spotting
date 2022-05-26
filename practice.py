import pickle
import numpy as np
from collections import Counter

with open('MSSL_TRAIN_SET_GT.pkl', 'rb') as f:
        data = pickle.load(f)

classes= []


# keys are files so iterate only limited files due to memory limitations.
for key in list(data.keys()):
    filename = key
    # file functions

    for x in data[key]:
        classes.append(x[0])
        # start_time = x[1]
        # end_time = x[2]
        #diff = x[2] - x[1]
        


x = dict(Counter(classes))
count = dict(sorted(x.items(), key=lambda item: item[1]))

print("Count of classes:")
print(x,"\n")
time = dict.fromkeys(x.keys(),0)
# print(time)


for key in list(data.keys()):
    filename = key
    # file functions

    for x in data[key]:
        #classes.append(x[0])
        diff = x[2] - x[1]
        # print(x[0])
        #time[x[0]].append(diff)
        if time[x[0]] < diff:
            time[x[0]] = diff
print("Longest Duration of each class:")
#print(dict(sorted(time.items(), key=lambda item: item[1])))

print(time)

for key in list(data.keys()):
    filename = key
    # file functions

    for x in data[key]:
        #classes.append(x[0])
        diff = x[2] - x[1]
        # print(x[0])
        #time[x[0]].append(diff)
        if time[x[0]] > diff:
            time[x[0]] = diff
print("Shortest Duration of each class:")
print(time)



