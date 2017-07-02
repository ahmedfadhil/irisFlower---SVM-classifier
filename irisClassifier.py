import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# iris = sns.load_dataset('iris')
iris = pd.read_csv('Iris.csv')
iris.head()
iris.info()

sns.pairplot(iris, hue='Species', palette='Dark2')

setosa = iris[iris['Species'] == 'Iris-setosa']

sns.kdeplot(setosa['SepalWidthCm'], setosa['SepalLengthCm'], cmap='plasma', shade=True, shade_lowest=False)

# Split the data into train and test

X = iris.drop('Species', axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

svc_mode = SVC()
svc_mode.fit(X_train, y_train)
predictions = svc_mode.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}

grid = GridSearchCV(SVC(), param_grid, verbose=2)

grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
