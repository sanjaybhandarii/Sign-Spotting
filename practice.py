import pickle

with open('predictions.pkl', 'rb') as f:
        data = pickle.load(f)

for key in list(data.keys())[9:10]:

        for x in data[key]:
            print(x)
          
   #give path
