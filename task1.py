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
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm.autonotebook import tqdm

# spacy.cli.download("en_core_web_sm")
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

# time_str = f"\t{datetime.datetime.now().strftime('%H:%M:%S')}"

__all__ = ['tokens', 'preprocessing']


def ifprint(s, **kwargs):
    if verbose: print(s, **kwargs)


def load_tqdm(iterate, **kwargs):
    return tqdm(iterate, **kwargs) if verbose else iterate


def get_tokens(text) -> list:
    # words = r'?:\w+[\']?\w'
    # numbers = r'?:[-]?\d+(?:[,-.]\d+'
    # symbols = r'?:[\\(){}[\]=&|^+<>/*%;\.,"\'?!~——]'
    #
    # token_pattern = fr'({symbols})|({numbers})*|({words}))'
    # tokens = re.findall(token_pattern, string=text)
    docs = ['']
    paragraphs = text.split('\n')
    i = 0
    count = 0
    for p in load_tqdm(paragraphs):
        count += len(p)
        if count >= 100000:
            i += 1
            count = len(p)
            docs.append('')
        docs[i] += p

    nlp = spacy.load("en_core_web_sm")
    tokens = []
    for doc in load_tqdm(docs, unit='word', desc='tokenizing'):
        tokens += list(nlp(doc))

    return tokens


def read_txt(filename='./data/passage-collection.txt'):
    ifprint('complete', end='')
    with open(filename, "r", encoding='utf-8') as f:  # 打开文件
        data = f.read()  # 读取文件
    ifprint('complete')
    return data


def preprocessing(data):
    ifprint('complete', end='')
    data = get_tokens(data.lower())
    data = remove_nums(data)
    ifprint('complete')

    # 把一个任何形式的语言词汇还原为一般形式（能表达完整语义）
    lemmatizer = WordNetLemmatizer()
    data = [lemmatizer.lemmatize(d) for d in (load_tqdm(data, unit='word', desc='lemmatizing'))]
    return Counter(data).most_common()


def clean_tokens(word_list, remove_symbols=True, remove_nums=True):
    remove_list = '\\(){}[\]=&|^+<>/*%;.,\"\'?!~——'
    if remove_symbols:
        word_list = list(filter(lambda x: x not in remove_list, word_list))
    if remove_nums:
        word_list = list(filter(lambda x: re.match(r'[-]*(\d+[,-.]*\D*)+', x) is None, word_list))
    return word_list


def remove_stop_words(text):
    remove_words = stopwords.words('english')
    indexes = [i for i in range(text.shape[0]) if text[i, 0] not in remove_words]
    return text[indexes]


def plot_zipf(word_freq_list, remove=False):
    count_words = word_freq_list[:, 1].astype('int')
    num_words = np.sum(count_words)
    ifprint(f'Number of words: {num_words}')
    N = len(word_freq_list)
    ifprint(f'Vocabulary: {N}')
    result = count_words / num_words

    x = np.linspace(1, N, N)
    sum_i = sum([1 / i for i in range(1, N + 1)])
    ifprint(f'sum_i = {sum_i}')
    zipfLaw = np.array([1 / k for k in x]) / sum_i
    ifprint(f'MSE = {np.mean(np.square(result - zipfLaw))}')

    for str, func in zip(['', '(log)'], [plt.plot, plt.loglog]):
        func(x, result, label='data')
        func(x, zipfLaw, label="theory (Zipf's Law)")
        plt.xlabel('Term freq. ranking' + str)
        plt.ylabel('Term prob. of occurrence' + str)
        plt.legend()
        plt.savefig(f'freq_prob_plot{str}' + '_stopwords_removed' if remove else '')
        plt.show()


verbose = __name__ == '__main__'
vocab_file_name = 'vocab2.npy'

try:
    tokens = np.load(vocab_file_name)
except FileNotFoundError:
    tokens = np.array(preprocessing(read_txt()))
    np.save(vocab_file_name, tokens)

if __name__ == '__main__':
    plot_zipf(tokens)
    plot_zipf(remove_stop_words(tokens), True)
