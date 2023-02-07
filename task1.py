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
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import nltk

nltk.download('stopwords')
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from tqdm.autonotebook import tqdm


def get_tokens(text) -> list:
    words = r'?:\w+[\']?\w'
    numbers = r'?:[-]?\d+(?:[,-.]\d+'
    symbols = r'?:[\\(){}[\]=&|^+<>/*%;\.,"\'?!~——]'

    token_pattern = fr'({symbols})|({numbers})*|({words}))'
    tokens = re.findall(token_pattern, string=text)

    return tokens


def read_txt(filename='./data/passage-collection.txt'):
    print('reading txt')
    with open(filename, "r", encoding='utf-8') as f:  # 打开文件
        data = f.read()  # 读取文件
    return data


def preprocessing():
    data = read_txt()
    print('tokenizing')
    data = get_tokens(data)
    data = del_symbols(data)
    print('stemming')
    stemmer = PorterStemmer()
    data = [stemmer.stem(d) for d in tqdm(data)]
    print('lemmatizing')
    lemmatizer = WordNetLemmatizer()
    data = [lemmatizer.lemmatize(d) for d in tqdm(data)]
    print(f'Vocabulary: {len(data)}')
    return data


def del_symbols(word_list):
    remove_list = '\\(){}[\]=&|^+<>/*%;.,\"\'?!~——'
    b = filter(lambda x: x not in remove_list, word_list)
    return list(b)


def remove_stop_words(text):
    remove_words = stopwords.words('english')
    result = filter(lambda w: w not in remove_words, text)
    return list(result)


def plot_zipf(tokens):
    sort_hist = Counter(tokens).most_common()
    num_words = len(tokens)
    print(f'Number of words: {num_words}')
    N = len(sort_hist)
    print(f'Vocabulary: {N}')
    result = np.array([time / num_words for word, time in sort_hist])

    x = np.linspace(1, N, N)
    sum_i = sum([1 / i for i in range(1, N + 1)])
    print(f'sum_i = {sum_i}')
    zipfLaw = np.array([1 / k for k in x]) / sum_i
    print(f'MSE = {np.mean(np.square(result - zipfLaw))}')

    for str, func in zip(['', '(log)'], [plt.plot, plt.loglog]):
        func(x, result, label='data')
        func(x, zipfLaw, label='theory')
        plt.xlabel('Term freq. ranking' + str)
        plt.ylabel('Term prob. of occurrence' + str)
        plt.legend()
        plt.show()


save = False
if __name__ == '__main__':

    if save:
        tokens = preprocessing()
        with open('new.json', 'w') as f:
            json.dump(tokens, f)
    else:
        with open('new.json', 'r') as f:
            tokens = json.load(f)

    plot_zipf(tokens)
    plot_zipf(remove_stop_words(tokens))
