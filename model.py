import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = {
    'fever': [1,0,1,0,1],
    'cough': [1,1,0,0,1],
    'headache': [1,0,1,0,1],
    'fatigue': [1,1,0,0,1],
    'vomiting': [0,0,1,0,1],
    'cold': [1,1,0,1,1],
    'disease': ['Flu','Cold','Migraine','Healthy','Flu']
}

df = pd.DataFrame(data)

X = df[['fever','cough','headache','fatigue','vomiting','cold']]
y = df['disease']

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))