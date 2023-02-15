#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import collections

import pandas as pd
from tqdm.auto import tqdm

from task2 import InvertedIndex, passages_indexes, tokens, preprocessing


# Extract the TF-IDF vector representations of the passages using the inverted index you
# have constructed.

# Using the IDF representation from the corpus of the passages,
# extract the TF-IDF representation of the queries.

# Use a basic vector space model with TF-IDF and cosine similarity
# to retrieve at most 100 passages from within the 1000 candidate passages

# for each query (some queries have fewer candidate passages).
# Store the outcomes in a file named tfidf.csv


def generate_queries_indexes(data_location='./data/test_queries.tsv'):
    queries = pd.read_csv(data_location,
                          sep='\t', header=None,
                          names=['qid', 'content'],
                          index_col=[0]).drop_duplicates()

    queries_indexes = InvertedIndex(initial_tokens=tokens)

    for qid, query in tqdm(queries.iterrows(), total=queries.shape[0]):
        a = preprocessing(query.content)
        a = collections.Counter(a).most_common()
        [queries_indexes.add_word(word, document=qid, count=count) for word, count in a]

    queries_indexes.idf_reference = passages_indexes
    return queries_indexes


if __name__ == '__main__':
    tfidf_passages = passages_indexes.get_all_feature_matrix(to_dataframe=True)
    queries_indexes = generate_queries_indexes()
    tfidf_queries = queries_indexes.get_all_feature_matrix(True)
