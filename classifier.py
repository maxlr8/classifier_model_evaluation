# pylint: disable-all

# Importing Libraries

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report 

warnings.filterwarnings('ignore')

# Importing Dataset

path = os.path.join(os.getcwd(), r'dataset\breast-cancer.csv')
data = open(path)
df = pd.read_csv(data)

# Encoding

df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0}).astype(int)

# Removal of redundant and highly-correlated columns

df = df.drop(['id'], axis=1)
df = df.drop(["radius_mean", "texture_mean", "area_worst", "perimeter_worst", "concave points_worst", "compactness_worst", "area_se", "perimeter_se", "concave points_se", "compactness_se", "area_mean", "perimeter_mean", "concave points_mean", "concavity_mean"], axis=1)

# Correlation Matrix

corr = df.corr()
plt.figure(figsize=(30,10))
sns.heatmap(corr, cmap="Greens",annot=True)
plt.show()

# Dataset splitting

X = df.iloc[ : , 1:]
Y = df.iloc[ : , :1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1) # 10 samples for test

# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LogisticRegression(random_state = 1)
model.fit(X_train, Y_train.values.ravel())
predict = model.predict(X_test)

# Logistic Regression

print("\nLogistic Regression\n")

print(f"Accuracy Score: {accuracy_score(predict, Y_test)*100:.2f}%\n")

cm = confusion_matrix(Y_test, predict)
print('''Confusion Matrix:\n''')
print(cm)

print(f"\nClassification Report:\n{classification_report(Y_test, predict)}")

model = GaussianNB()
model.fit(X_train, Y_train.values.ravel())
predict = model.predict(X_test)

# Naive Bayes Classifier

print("\nNaive Bayes Classifier\n")

print(f"Accuracy Score: {accuracy_score(predict, Y_test)*100:.2f}%\n")

cm = confusion_matrix(Y_test, predict)
print('''Confusion Matrix:\n''')
print(cm)

print(f"\nClassification Report:\n{classification_report(Y_test, predict)}")

model = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2)
model.fit(X_train, Y_train.values.ravel())
predict = model.predict(X_test)

# K-Nearest Neighbours

print("\nK-Nearest Neighbours\n")

print(f"Accuracy Score: {accuracy_score(predict, Y_test)*100:.2f}%\n")

cm = confusion_matrix(Y_test, predict)
print('''Confusion Matrix:\n''')
print(cm)

print(f"\nClassification Report:\n{classification_report(Y_test, predict)}")

model = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
model.fit(X_train, Y_train.values.ravel())
predict = model.predict(X_test)

# Decision Tree Classifier

print("\nDecision Tree Classifier\n")

print(f"Accuracy Score: {accuracy_score(predict, Y_test)*100:.2f}%\n")

cm = confusion_matrix(Y_test, predict)
print('''Confusion Matrix:\n''')
print(cm)

print(f"\nClassification Report:\n{classification_report(Y_test, predict)}")

model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 1)
model.fit(X_train, Y_train.values.ravel())
predict = model.predict(X_test)

# Random Forst Classifier

print("\nRandom Forst Classifier\n")

print(f"Accuracy Score: {accuracy_score(predict, Y_test)*100:.2f}%\n")

cm = confusion_matrix(Y_test, predict)
print('''Confusion Matrix:\n''')
print(cm)

print(f"\nClassification Report:\n{classification_report(Y_test, predict)}")

model = SVC(kernel = 'rbf', random_state = 1)
model.fit(X_train, Y_train.values.ravel())
predict = model.predict(X_test)

# Support Vector Machines

print("\nSupport Vector Machines\n")

print(f"Accuracy Score: {accuracy_score(predict, Y_test)*100:.2f}%\n")

cm = confusion_matrix(Y_test, predict)
print('''Confusion Matrix:\n''')
print(cm)

print(f"\nClassification Report:\n{classification_report(Y_test, predict)}")

'''
[ TRUE NEGATIVES] [FALSE POSITIVE]
[FALSE NEGATIVES] [ TRUE POSITIVE]
'''