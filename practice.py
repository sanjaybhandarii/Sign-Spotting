import pickle

classes = []

with open('MSSL_TRAIN_SET_GT.pkl', 'rb') as f:
    data = pickle.load(f)



y = data['p01_n002']
print(y[0])
# for x in y[0]:
#     print(x)

for key in data.keys():
    filename = key
    print("file",filename)

    # file functions

    for x in data[key]:
        classes.append(x[0])
        start_time = x[1]
        end_time = x[2]
#         for y in x:
#             print(y)
            # classes.append(int(x[0]))
            # start_time = x[1]
            # end_time = x[2]