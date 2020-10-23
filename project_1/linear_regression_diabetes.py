#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, metrics, linear_model, model_selection
from scipy import stats


def main():
	diabetes = datasets.load_diabetes()
	x = diabetes.data[:, np.newaxis, 2]
	y = diabetes.target
	linear_regression_model = linear_model.LinearRegression()

	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)

	regressor = linear_regression_model.fit(x_train, y_train)
	slope = regressor.coef_
	intercept = regressor.intercept_

	y_predicted = linear_regression_model.predict(x_test)

	print("Correlation: %.2f" % stats.pearsonr(y_test, y_predicted)[0])
	print("Mean Squared Error: %.2f" % metrics.mean_squared_error(y_test, y_predicted))
	print("R^2: %.2f" % metrics.r2_score(y_test, y_predicted))

	plt.figure(1, figsize=(6, 5))
	plt.plot(x, slope*x + intercept, 'r', label='y={:.2f}x+{:.2f}'.format(slope[0], intercept))
	plt.scatter(x, y, color='k', s=6)
	plt.legend(fontsize=9)

	plt.show()



if __name__ == '__main__':
	main()
