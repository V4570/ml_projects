#!/usr/bin/env python3

"""
AI101 Machine Learning
Project 6.1 - Leave-One-Out Cross-Validation
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut


def main():
    df = pd.read_csv('biomechanical_features.csv')

    x = df.drop('class', axis=1).to_numpy()
    y = df['class'].to_numpy()

    loo = LeaveOneOut()

    sc = []
    knn = KNeighborsClassifier(n_neighbors=5)
    for train, test in loo.split(x):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        knn.fit(x_train, y_train)
        y_predicted = knn.predict(x_test)

        acc = metrics.accuracy_score(y_test, y_predicted)
        sc.append(acc)

    y_pred = knn.predict(x)

    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()

    print(f'Average accuracy: {sum(sc)/len(sc):.3f}')
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')


if __name__ == '__main__':
    main()
