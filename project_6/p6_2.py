#!/usr/bin/env python3

"""
AI101 Machine Learning
Project 6.2 - Friedman test
"""
import pandas as pd
from scipy.stats import friedmanchisquare


def main():
    df = pd.read_csv('algo_performance.csv')

    s, p = friedmanchisquare(df['C4.5'], df['1-NN'], df['NaiveBayes'], df['Kernel'], df['CN2'])

    a = 0.05
    while a > 0.0:
        print(f'a: {a:.2f}, p: {p:.3e}')
        if p > a:
            print('Distributions are the same.')
        else:
            print('At least one has a different distribution.')

        a -= 0.01
    print(f"Statistics = {s:.3f}, p={p:.3e}")


if __name__ == '__main__':
    main()
