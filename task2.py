#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Using the vocabulary of terms identified in Task 1 (you will need to choose between removing or keeping stop words)
Build an inverted index for the collection so that you can retrieve passages in an efficient way.
"""
import os.path
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from tqdm.auto import tqdm

from task1 import preprocessing, tokens

# __all__ = ['passages_indexes', 'InvertedIndex', 'task1']

vocab_size = len(tokens)
vocab_dict = dict(zip(tokens[:, 0], range(vocab_size)))


class InvertedIndex:
    def __init__(self, initial_tokens=tokens):
        self.freq_mat = np.zeros((vocab_size, vocab_size))
        self.idf_reference = None

    def add_word(self, paragraphs, count=1):
        """
        Update index(es) for the word in the specified document.
        """
        # for pid,p in paragraphs:

    def word_freq_doc(self, word, document, normalized=True):
        """
        Returns the frequency of the word in the specified document.
        """
        result = self._words[word].get(document, 0)
        if normalized:
            result /= self.total_words_doc(document)

        return float(result)

    def total_words_doc(self, document):
        """
        Returns the number of words in the specified document.
        """
        return self._documents[document]

    def word_freq_corpus(self, word, normalized=True):
        """
        Return the frequency of the specified word in the corpus (word collections)
        """
        result = sum(self._words[word].values())
        if normalized:
            result /= sum(self._documents.values())
        return result

    def count_word_appear_doc(self, word):
        """
        Returns the number of documents the specified word appears in.
        """
        return len(self._words[word])

    def get_idf(self, word):
        """
        Inverse-Document-Frequency value for the given word
        """
        if self.idf_reference is None:
            df = self.count_word_appear_doc(word)  # number of documents the word appears in
            N = len(self._documents)  # number of documents in collection
            return np.log10(N / df)

        else:
            return self.idf_reference.get_idf(word)

    def get_tfidf(self, word, document):
        """
        Returns the Term-Frequency Inverse-Document-Frequency value for the given
        word in the specified document.
        """
        tf = self.word_freq_doc(word, document)
        if tf == 0.0:
            return 0.0
        return tf * self.get_idf(word)

    def get_doc_feature_array(self, document):

        result = self._words[word]
        if normalized:
            result /= self.total_words_doc(document)

        return [self.get_tfidf(word, document) for word in self._words]

    def get_all_feature_matrix(self, to_dataframe=False):
        """
        Returns the feature matrix.

        Size of the matrix: m x n
        m: # documents
        n: vocabulary size
        """
        result = [self.get_doc_feature_array(document) for document in tqdm(self._documents, desc='Generating Matrix')]
        if to_dataframe:
            result = pd.DataFrame(result)
            result.columns = list(self._words.keys())
            result.index = list(self._documents.keys())
        return result


def read_data(data_location='./data/candidate-passages-top1000.tsv'):
    df = pd.read_csv(data_location,
                     sep='\t', header=None, usecols=[1, 3],
                     names=['pid', 'content']).drop_duplicates()
    return df.reset_index(drop=True)


def generate_indexes(passages_dataframe):
    passages_len = passages_dataframe.shape[0]
    inverted_indexes = lil_matrix((vocab_size, passages_len))

    for index, passage in tqdm(passages_dataframe.iterrows(),
                               total=passages_len, desc='Invert indexing', unit='passage'):
        word_counter = preprocessing(passage.content)
        word_counter = Counter(word_counter)
        for word, count in word_counter:
            inverted_indexes[vocab_dict[word], index] = count

    return inverted_indexes


file_name = 'inverted_index.pkl'
if os.path.exists(file_name):
    with open(file_name, 'rb') as file:
        passages_indexes = pickle.load(file)
else:
    passages_df = read_data()
    passages_indexes = generate_indexes(passages_df)
    with open(file_name, 'wb') as file:
        pickle.dump(passages_indexes, file)

if __name__ == '__main__':
    X = passages_indexes.get_all_feature_matrix(to_dataframe=True)
