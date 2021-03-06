import cv2
import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

ls = list()

#print(os.path.dirname(os.path.realpath(__file__)))
pos_fold = "C:/Users/Sarper/Desktop/software/drone_pics"
neg_fold = "C:/Users/Sarper/Desktop/software/other_pics"

for filename in os.listdir(pos_fold):
    img = cv2.imread(os.path.join(pos_fold, filename), 0)
    if img is not None:
        img = cv2.resize(img, (300, 300), cv2.INTER_AREA)
        ls.append(img.ravel())

for filename in os.listdir(neg_fold):
    img = cv2.imread(os.path.join(neg_fold, filename), 0)
    if img is not None:
        img = cv2.resize(img, (300, 300), cv2.INTER_AREA)
        ls.append(img.ravel())

x = np.array(ls)
y = np.array(len(os.listdir(pos_fold)) * [1] + len(os.listdir(neg_fold)) * [0]).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

mlp = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=5000)
mlp.fit(x_train, y_train.ravel())

y_pred = mlp.predict(x_test)

z = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_pred)], axis=1)
z.columns = ['True', 'Prediction']
print(z)

accuracy = accuracy_score(y_test, y_pred)

print("Tahminin Doğruluk Oranı: %", accuracy*100)

print(classification_report(y_test, y_pred))
