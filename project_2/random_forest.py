#!/usr/bin/env python3

"""
AI101 Machine Learning
Project 2 - Random Forest
"""

__author__ = "Vasileios Tosounidis"
__email__ = "vtosounid@csd.auth.gr"

from sklearn import datasets, metrics, ensemble, model_selection
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')


def main():
	breast_cancer_data = datasets.load_breast_cancer()

	num_of_features = 5

	x = breast_cancer_data.data[:, :num_of_features]
	y = breast_cancer_data.target

	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)

	criterion = ['gini', 'entropy']
	max_depth = 3
	n_estimators = 1

	results = {'ac': [], 'pr': [], 're': [], 'f1': []}

	print('Criterion,Number of Estimators,Accuracy,Precision,Recall,F1')
	for c in criterion:
		for n in range(n_estimators, 201):
			model = ensemble.RandomForestClassifier(criterion=c, n_estimators=n, max_depth=max_depth)
			model.fit(x_train, y_train)

			y_predicted = model.predict(x_test)

			accuracy = metrics.accuracy_score(y_test, y_predicted)
			results['ac'].append(accuracy)

			precision = metrics.precision_score(y_test, y_predicted)
			results['pr'].append(precision)

			recall = metrics.recall_score(y_test, y_predicted)
			results['re'].append(recall)

			f1 = metrics.f1_score(y_test, y_predicted)
			results['f1'].append(f1)

			print('{},{},{:.2f},{:.2f},{:.2f},{:.2f}'.format(c, n, accuracy, precision, recall, f1))

	fig, axs = plt.subplots(2, 4, figsize=(55, 30))
	fig.suptitle("Results split depending on criterion option\n", fontsize=50)
	plt.figtext(0.25, 0.93, "Gini\n", va="center", ha="center", size=35)
	plt.figtext(0.75, 0.93, "Entropy\n", va="center", ha="center", size=35)

	axs[0, 0].plot(list(range(n_estimators, 201)), results['ac'][:200])
	axs[0, 0].set_title('Accuracy')
	axs[0, 1].plot(list(range(n_estimators, 201)), results['pr'][:200], 'tab:orange')
	axs[0, 1].set_title('Precision')
	axs[1, 0].plot(list(range(n_estimators, 201)), results['re'][:200], 'tab:green')
	axs[1, 0].set_title('Recall')
	axs[1, 1].plot(list(range(n_estimators, 201)), results['f1'][:200], 'tab:red')
	axs[1, 1].set_title('F1')

	axs[0, 2].plot(list(range(n_estimators, 201)), results['ac'][200:])
	axs[0, 2].set_title('Accuracy')
	axs[0, 3].plot(list(range(n_estimators, 201)), results['pr'][200:], 'tab:orange')
	axs[0, 3].set_title('Precision')
	axs[1, 2].plot(list(range(n_estimators, 201)), results['re'][200:], 'tab:green')
	axs[1, 2].set_title('Recall')
	axs[1, 3].plot(list(range(n_estimators, 201)), results['f1'][200:], 'tab:red')
	axs[1, 3].set_title('F1')
	plt.show()


if __name__ == '__main__':
	main()
