import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


data = pd.read_csv("breast-cancer.csv")

# print(data.head())
# print(data.shape)
# print(data.info())
# print(data.columns)
# print(data.describe())
# print(data.isnull().sum())
    
# print(data['diagnosis'].value_counts())


data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data['diagnosis']=data['diagnosis'].astype(int)
# print(data.head())

x = data.drop(columns=['id', 'diagnosis'], axis=1)
y = data['diagnosis']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

models = {
    'Linear SVM': SVC(kernel='linear'),
    'Polynomial SVM': SVC(kernel='poly', degree=3),
    'RBF SVM': SVC(kernel='rbf')
}
for name , model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    print(f"{name} Accuracy is : {acc*100:.2f}%")
print("\n") 
for d in [2, 3, 4, 5]:
    model = SVC(kernel='poly', degree=d)
    model.fit(x_train, y_train)
    svm_pred = model.predict(x_test)
    acc = accuracy_score(y_test, svm_pred)
    print(f"Degree {d} : Accuracy: {acc*100:.2f}%")
print("\n") 
for gama in [0.1, 1, 10, 100]:
    model = SVC(kernel='rbf', gamma=gama)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    print(f"Gamma {gama} : Accuracy: {acc*100:.2f}%")