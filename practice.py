import pickle

with open('predictions.pkl', 'rb') as f:
        data = pickle.load(f)

print(data[list(data.keys())[1]])
