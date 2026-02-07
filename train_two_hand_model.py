import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './two_hand_data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue
        
    for file_name in os.listdir(class_dir):
        if file_name.endswith('.pickle'):
            file_path = os.path.join(class_dir, file_name)
            with open(file_path, 'rb') as f:
                data.append(pickle.load(f))
            labels.append(dir_)

X = np.asarray(data)
y = np.asarray(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)

print(f'Model Accuracy: {accuracy * 100:.2f}%')

with open('two_hand_model.p', 'wb') as f:
    pickle.dump(model, f)

print('Model saved as two_hand_model.p')