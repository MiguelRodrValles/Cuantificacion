# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digit=datasets.load_digits()

data=digit.data
target=digit.target

#La primera instrucción genera el marco en el que se va a establecer el gráfico,
# ncols hacer referencia al número de columnas (gráficos) (El operador _, sirve
# para desechar esa asignación, es decir sólo importa axes)
_, axes = plt.subplots(nrows=1, ncols=9, figsize=(10, 3))
for ax, image, label in zip(axes, digit.images, digit.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i ' % label)

#reshape permite restructurar la figura en las dimensiones adecuadas, el -1 quiere
# decir que se infiere a partir de la otra, este conjunto podia haberse tomado directamente
#del data
n_samples = len(digit.images)
data = digit.images.reshape((n_samples, -1))

#SVM
# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digit.target, test_size=0.5, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

# Cross Validation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
predicted = clf.predict(X_test)
clf.score(X_test, y_test)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
scores = cross_val_score(clf, X, y, cv=5)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))