import pickle

with open('MSSL_TRAIN_SET_GT.pkl', 'rb') as f:
    data = pickle.load(f)

length = 0
 
for x in data:
    length += len(data[x])

print(length)