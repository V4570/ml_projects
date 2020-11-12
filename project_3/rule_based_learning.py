#!/usr/bin/env python3

"""
AI101 Machine Learning
Project 3 - Rule-Based Learning
"""

__author__ = "Vasileios Tosounidis"
__email__ = "vtosounid@csd.auth.gr"

import Orange


def main():
	wine_data = Orange.data.Table("wine.csv")

	learners = [Orange.classification.CN2UnorderedLearner(), Orange.classification.CN2Learner()]

	testing_values = [(10, 10, 5), (2, 10, 5), (10, 2, 5), (10, 10, 10), (10, 2, 10), (2, 3, 1)]

	for learner in learners:
		print('Testing for learner: ', learner)

		if str(learner) == 'cn2':
			evaluators = [Orange.classification.rules.EntropyEvaluator(),
			              Orange.classification.rules.LaplaceAccuracyEvaluator()]

			for evaluator in evaluators:
				print("Evaluator: {}".format(str(evaluator)))
				for values in testing_values:
					print("Beam width: {}, Min rule coverage: {}, Max rule length: {}".format(values[0], values[1], values[2]))
					evaluate(learner, wine_data, evaluator, beam_width=values[0],
					                           min_covered_examples=values[1], max_rule_length=values[2])
		else:
			for values in testing_values:
				print("Beam width: {}, Min rule coverage: {}, Max rule length: {}".format(values[0], values[1], values[2]))
				evaluate(learner, wine_data, Orange.classification.rules.LaplaceAccuracyEvaluator(),
				                           beam_width=values[0], min_covered_examples=values[1],
				                           max_rule_length=values[2])


def evaluate(learner, wine_data, evaluator, beam_width=10, min_covered_examples=15, max_rule_length=2):
	learner.rule_finder.search_algorithm.beam_width = beam_width
	learner.rule_finder.general_validator.min_covered_examples = min_covered_examples
	learner.rule_finder.general_validator.max_rule_length = max_rule_length

	learner.rule_finder.quality_evaluator = evaluator

	cv = Orange.evaluation.CrossValidation()
	results = cv(wine_data, [learner])

	ca = Orange.evaluation.CA(results)[0]
	f1 = Orange.evaluation.F1(results, average='macro')[0]
	rec = Orange.evaluation.Recall(results, average='macro')[0]
	pr = Orange.evaluation.Precision(results, average='macro')[0]

	print('------Scores------')
	print("\tPrecision: {:.3f}".format(pr))
	print("\tRecall: {:.3f}".format(rec))
	print("\tF1: {:.3f}".format(f1))
	print("\tAccuracy: {:.3f}".format(ca))

	classifier = learner(wine_data)

	print("------Rules-------")
	for rule in classifier.rule_list:
		print("\t{}".format(rule))


if __name__ == '__main__':
	main()
