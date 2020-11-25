#!/usr/bin/env python3

"""
AI101 Machine Learning
Project 5 - Bayesian Learning
"""

__author__ = "Vasileios Tosounidis"
__email__ = "vtosounid@csd.auth.gr"

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    train_news_dataset = fetch_20newsgroups(subset='train',
                                            remove=('headers', 'footers', 'quotes'),
                                            shuffle=False, random_state=0)
    test_news_dataset = fetch_20newsgroups(subset='test',
                                           remove=('headers', 'footers', 'quotes'),
                                           shuffle=False, random_state=0)
    vectorizer = CountVectorizer()

    x_train = vectorizer.fit_transform(train_news_dataset.data)
    y_train = train_news_dataset.target

    x_test = vectorizer.transform(test_news_dataset.data)
    y_test = test_news_dataset.target

    acc = 0.0
    recall = 0.0
    precision = 0.0
    f1 = 1.0
    a = 0.1
    y_predicted = None
    while f1 >= 0.7:
        classifier = MultinomialNB(alpha=a)
        classifier.fit(x_train, y_train)

        y_predicted = classifier.predict(x_test)
        acc = accuracy_score(y_test, y_predicted)
        recall = recall_score(y_test, y_predicted, average='macro')
        precision = precision_score(y_test, y_predicted, average='macro')
        f1 = f1_score(y_test, y_predicted, average='macro')
        a += 0.05

    a -= 0.05
    cf_matrix = confusion_matrix(y_test, y_predicted)
    print('------Scores------')
    print("\tAccuracy: {:.3f}".format(acc))
    print("\tPrecision: {:.3f}".format(precision))
    print("\tRecall: {:.3f}".format(recall))
    print("\tF1: {:.3f}".format(f1))

    plt.figure(figsize=(20, 10))
    title = 'Multinomial NB - Confusion Matrix (a = {:.2f}) ' \
            '[Acc = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}]'.format(a, acc, precision, recall, f1)
    plt.title(title)
    df_cf = pd.DataFrame(cf_matrix, index=test_news_dataset.target_names, columns=test_news_dataset.target_names)

    heatmap = sns.heatmap(df_cf, annot=True, fmt='d')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)

    plt.show()


if __name__ == '__main__':
    main()
