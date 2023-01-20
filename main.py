"""
Task 1 – Text statistics (30 marks). Use the data in passage-collection.txt for this
task. Extract terms (1-grams) from the raw text. In doing so, you can also perform basic text
preprocessing steps. However, you can also choose not to.



Do not remove stop words in the first instance.

Describe and justify any other text preprocessing choice(s), if any, and report the size of the identified index of
terms (vocabulary).

Then, implement a function that counts the number of occurrences of terms in the provided data set,  plot their
probability of occurrence (normalised frequency) against their frequency ranking,  and qualitatively  justify  that
these  terms  follow  Zipf ’s  law
"""

import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def get_tokens(text) -> list:
    words = r'?:\w+[\']?\w'
    numbers = r'?:[-]?\d+(?:[,-.]\d+'
    symbols = r'?:[\\(){}[\]=&|^+<>/*%;\.,"\'?!~——]'

    token_pattern = fr'({symbols})|({numbers})*|({words}))'
    print('tokenizing')
    tokens = re.findall(token_pattern, string=text)

    return tokens


def read_txt(filename='./data/passage-collection.txt'):
    print('reading txt')
    with open(filename, "r", encoding='utf-8') as f:  # 打开文件
        data = f.read()  # 读取文件
    return data


def preprocessing():
    data = read_txt()
    words = get_tokens(data)
    return delnum(words)


def delnum(word_list):
    remove_list = '\\(){}[\]=&|^+<>/*%;.,\"\'?!~——'
    b = filter(lambda x: x not in remove_list, word_list)
    return list(b)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    tokens = preprocessing()
    sort_hist = Counter(tokens).most_common()
    num_words = len(tokens)
    N = len(sort_hist)
    result = [time / num_words for word, time in sort_hist]

    x = np.linspace(1, N, N)
    sum_i = sum([1 / i for i in range(1, N + 1)])
    ZipfLaw = np.array([1 / k for k in x]) / sum_i

    for str, func in zip(['', '(log)'], [plt.plot, plt.loglog]):
        func(x, result, label='data')
        func(x, ZipfLaw, label='theory')
        plt.xlabel('Term freq. ranking' + str)
        plt.ylabel('Term prob. of occurrence' + str)
        plt.legend()
        plt.show()
