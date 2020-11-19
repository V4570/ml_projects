#!/usr/bin/env python3

"""
AI101 Machine Learning
Project 4 - Instance-based Learning
"""

__author__ = "Vasileios Tosounidis"
__email__ = "vtosounid@csd.auth.gr"

import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
import random
import matplotlib.pyplot as plt

random.seed = 42
np.random.seed(666)


def main():
	df = pd.read_csv('titanic.csv')

	# This was used to determine the correlation between individual features and the Survived feature.
	# The features with the highest correlation were chosen.
	# > print(abs(df.corr()['Survived']))

	# Below command showed a strong correlation between Fare and Pclass so only Pclass was retained since it has
	# a higher correlation to the Survived feature.
	# > print(abs(df.corr()[['Fare', 'Pclass']])

	# Replacing categorical values to numerical.
	df['Sex'].replace(['female', 'male'], [0, 1],  inplace=True)

	# Dropping unneeded features after above steps.
	df_no_impute = df.drop(['PassengerId', 'Name', 'SibSp', 'Ticket', 'Age', 'Cabin', 'Embarked', 'Fare', 'Parch'], axis=1)
	df_impute = df.drop(['PassengerId', 'Name', 'SibSp', 'Ticket', 'Cabin', 'Embarked', 'Fare', 'Parch'], axis=1)

	x = df_no_impute.drop('Survived', axis=1)
	y = df_no_impute.Survived

	x_impute = df_impute.drop('Survived', axis=1)
	y_impute = df_impute.Survived

	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)

	x_train_impute, x_test_impute, y_train_impute, y_test_impute = model_selection.\
		train_test_split(x_impute, y_impute, test_size=0.25, random_state=0)

	print('No impute')
	f1_no_impute = knn_train(x_train, x_test, y_train, y_test, False)

	print('Impute')
	f1_impute = knn_train(x_train_impute, x_test_impute, y_train_impute, y_test_impute, True)

	plt.figure(figsize=(10, 10))
	plt.title("k-Nearest Neighbors (Weights = 'distance', Metric = 'Minkowski', p = 2)")
	plt.plot(f1_impute, label='with impute')
	plt.plot(f1_no_impute, label='without impute')
	plt.legend()
	plt.xlabel('Number of neighbors')
	plt.ylabel('F1')
	plt.show()


def knn_train(x_train, x_test, y_train, y_test, impute):
	scaler = MinMaxScaler()
	scaler.fit(x_train, y_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	if impute:
		imputer = KNNImputer(n_neighbors=3)
		imputer.fit(x_train, y_train)
		x_train = imputer.transform(x_train)
		x_test = imputer.transform(x_test)

	k_neighbors = list(range(1, 200))
	weights = ['uniform', 'distance']
	p_params = [2, 1, 5]

	selected_fs = []

	for p in p_params:
		for w in weights:
			acc = 0.0
			recall = 0.0
			precision = 0.0
			best_f1 = 0.0
			best_k = 0
			for k in k_neighbors:
				knn = KNeighborsClassifier(n_neighbors=k, weights=w, p=p)
				knn.fit(x_train, y_train)
				y_predicted = knn.predict(x_test)

				f1 = metrics.f1_score(y_test, y_predicted, average='macro')
				if p == 2 and w == 'distance':
					selected_fs.append(f1)

				if f1 > best_f1:
					best_k = k
					best_f1 = f1
					acc = metrics.accuracy_score(y_test, y_predicted)
					recall = metrics.recall_score(y_test, y_predicted, average='macro')
					precision = metrics.precision_score(y_test, y_predicted, average='macro')

			print('Weight={}, p={}, accuracy={:.3f}, recall={:.3f}, precision={:.3f}, best_f1={:.3f}, best_k={}'.format(
				w, p, acc, recall, precision, best_f1, best_k
			))

	return selected_fs


if __name__ == '__main__':
	main()
