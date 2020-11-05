#!/usr/bin/env python3

"""
AI101 Machine Learning
Project 2 - Decision Tree
"""

__author__ = "Vasileios Tosounidis"
__email__ = "vtosounid@csd.auth.gr"

from sklearn import datasets, metrics, model_selection, tree
import graphviz


def main():
	breast_cancer_data = datasets.load_breast_cancer()

	num_of_features = 5

	x = breast_cancer_data.data[:, :num_of_features]
	y = breast_cancer_data.target

	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)
	criterion = ['gini', 'entropy']
	max_depth = 3

	print('Criterion,Max Depth,Accuracy,Precision,Recall,F1')
	for c in criterion:
		for md in range(max_depth, 15):
			model = tree.DecisionTreeClassifier(criterion=c, max_depth=md)

			model.fit(x_train, y_train)

			y_predicted = model.predict(x_test)

			accuracy = metrics.accuracy_score(y_test, y_predicted)
			precision = metrics.precision_score(y_test, y_predicted)
			recall = metrics.recall_score(y_test, y_predicted)
			f1 = metrics.f1_score(y_test, y_predicted)

			print('{},{},{:.2f},{:.2f},{:.2f},{:.2f}'.format(c, md, accuracy, precision, recall, f1))

			if md == 4:
				feature_names = breast_cancer_data.feature_names[:num_of_features]
				class_names = breast_cancer_data.target_names
				dot_data = tree.export_graphviz(model, feature_names=feature_names, class_names=class_names)

				graph = graphviz.Source(dot_data)
				graph.render("breastCancerTreePlot_" + c)


if __name__ == '__main__':
	main()
