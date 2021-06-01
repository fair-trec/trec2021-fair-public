"""
Implementation of metrics for TREC Fair Ranking 2021.
"""

import numpy as np


def qr_ndcg(qrun, qrels, tgt_n=100):
    """
    Compute the nDCG metric for Task 1.

    Args:
        qrun(array-like):
            The (ranked) document identifiers returned for a single query.
        qrels(array-like):
            The relevant document identifiers for this query.
        tgt_n(int):
            The maximum number of documents in a ranking.
    Returns:
        float:
            The nDCG score for the ranking.
    """
    if len(qrun) > tgt_n:
        raise ValueError(f'too many documents in query run')

    # length of run
    n = len(qrun)
    # max length of ideal run
    rel_n = min(tgt_n, len(qrels))

    # compute 1/0 utility scores
    util = np.isin(qrun, qrels).astype('f4')
    # compute discounts
    disc = np.log2(np.arange(1, tgt_n + 1, dtype='f4'))
    disc[0] = 1  # reset log of first
    disc = np.reciprocal(disc)

    # compute nDCG
    run_dcg = np.sum(util * disc[:n])
    ideal_dcg = np.sum(disc[:rel_n])

    return run_dcg / ideal_dcg