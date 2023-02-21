#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from scipy.sparse.sparsetools import csr_scale_rows

from task2 import generate_indexes, read_all_csv, passages_indexes, passages_df


def read_queries_csv(data_location='data/test-queries.tsv'):
    return pd.read_csv(data_location,
                       sep='\t', header=None,
                       names=['qid', 'content']
                       ).drop_duplicates().reset_index(drop=True)


def scale_csr(mat, scaler):
    result = mat.tocsr(copy=True)
    csr_scale_rows(result.shape[0],
                   result.shape[1],
                   result.indptr,
                   result.indices,
                   result.data, scaler)
    return result


def get_tf(inverted_indexes):
    count = inverted_indexes.sum(axis=1)
    return scale_csr(inverted_indexes, 1 / count)


def generate_tf_idf(tf, idf):
    return scale_csr(tf, idf)


def get_idf(inverted_indexes, log_func=np.log10, add_half=False):
    n = inverted_indexes.shape[1]
    non_zeros = inverted_indexes.indptr[1:] - inverted_indexes.indptr[:-1]
    return log_func((n - non_zeros + .5) / (non_zeros + .5)) if add_half else log_func(n / non_zeros)


def cosine_similarity(mat1: csr_matrix, mat2: csr_matrix):
    result = (mat1.T @ mat2).toarray()
    norm_1 = norm(mat1, axis=0).reshape(-1, 1)
    norm_2 = norm(mat2, axis=0).reshape(1, -1)

    return result / (norm_1 * norm_2)


def select_first100(scores):
    result = np.zeros((200 * 100, 3)) * np.nan
    for i in range(scores.shape[0]):
        qid = queries_dataframe.loc[i].qid
        candidates_pids = candidates_passages_df[candidates_passages_df.qid == qid].pid.values
        candidates_pids_idxs = passages_df[passages_df.pid.isin(candidates_pids)].index.values

        score = scores[i, candidates_pids_idxs]

        first_100 = np.argsort(score)[::-1][:100]
        indexes = first_100[score[first_100] > 0]

        pids = candidates_pids[indexes]

        result_idx = i * 100
        result[result_idx:result_idx + 100, 0] = qid
        result[result_idx:result_idx + len(indexes), 1] = pids
        result[result_idx:result_idx + len(indexes), 2] = score[indexes]

    result = pd.DataFrame(result, columns=['qid', 'pid', 'score']).dropna()
    result[['pid', 'qid']] = result[['pid', 'qid']].astype(int)
    return result


def get_p_length_normalized(inverted_indexes_p):
    doc_len = inverted_indexes_p.sum(axis=0)
    avdl = doc_len.mean()  # average document(passage) length
    return doc_len / avdl


def get_bm25(tf_p, tf_q, idf, p_len_normalized, k1=1.2, k2=100, b=.75):
    K = k1 * ((1 - b) + b * p_len_normalized)  # different for every passage

    temp0 = ((k1 + 1) * tf_p)
    temp1 = sparse_add_vec(tf_p, K)
    temp1.data = 1 / temp1.data
    S1 = temp0.multiply(temp1)

    left = scale_csr(S1, idf)

    temp0 = ((k2 + 1) * tf_q)
    temp1 = tf_q.copy()
    temp1.data = 1 / (temp1.data + k2)
    right = temp0.multiply(temp1)

    return left.T @ right


def sparse_add_vec(mat: csr_matrix, vec):
    vec = np.array(vec).reshape(-1)
    assert mat.shape[1] == vec.shape[0]
    mat = mat.tocsc()
    mat.data = mat.data * np.repeat(vec, mat.indptr[1:] - mat.indptr[:-1])
    return mat


if __name__ == '__main__':
    # 1. Extract IDF
    idf = get_idf(passages_indexes)

    # 2. Extract TF-IDF of passages
    passages_tf = get_tf(passages_indexes)
    passages_tfidf = generate_tf_idf(passages_tf, idf)

    # 3. Using idf_psgs, extract the TF-IDF of queries.
    queries_dataframe = read_queries_csv()
    queries_indexes = generate_indexes(queries_dataframe)
    queries_tf = get_tf(queries_indexes)
    queries_tfidf = generate_tf_idf(queries_tf, idf)

    # 4. Use a basic vector space model with TF-IDF and cosine similarity
    similarities = cosine_similarity(queries_tfidf, passages_tfidf)

    # 5. retrieve at most 100 passages from the 1000 passages for each query
    # no headers, expected to have 19,290 rows
    candidates_passages_df = read_all_csv()
    cosine_similarity_result = select_first100(similarities)

    # 6. Store the outcomes in a file named tfidf.csv
    cosine_similarity_result.to_csv('tfidf.csv', header=False, index=False)

    # 7. Use inverted index to implement BM25
    # while setting k1 = 1.2, k2 = 100, and b = 0.75.
    bm25_scores = get_bm25(tf_p=passages_tf, tf_q=queries_tf, idf=idf,
                           p_len_normalized=get_p_length_normalized(passages_indexes))

    # 8. Retrieve at most 100 passages from within the 1000 passages for each query.
    bm25_result = select_first100(bm25_scores)

    # 9. Store the outcomes in a file named bm25.csv
    bm25_result.to_csv('bm25.csv', header=False, index=False)
