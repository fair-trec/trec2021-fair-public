"""
Implementation of metrics for TREC Fair Ranking 2021.
"""

import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

_log = logging.getLogger('metrics')

world_pop = pd.Series({
    'Africa': 0.155070563,
    'Antarctica': 1.54424E-07,
    'Asia': 0.600202585,
    'Europe': 0.103663858,
    'Latin America and the Caribbean': 0.08609797,
    'Northern America': 0.049616733,
    'Oceania': 0.005348137,
})
work_order = [
    'Stub',
    'Start',
    'C',
    'B',
    'GA',
    'FA',
]

def discount(n_or_ranks):
    """
    Compute the discount function.

    Args:
        n_or_ranks(int or array-like):
            If an integer, the number of entries to discount; if an array,
            the ranks to discount.
    Returns:
        numpy.ndarray:
            The discount for the specified ranks.
    """
    if isinstance(n_or_ranks, int):
        n_or_ranks = np.arange(1, n_or_ranks + 1, dtype='f4')
    else:
        n_or_ranks = np.require(n_or_ranks, 'f4')
    n_or_ranks = np.maximum(n_or_ranks, 2)
    disc = np.log2(n_or_ranks)
    return np.reciprocal(disc)


def qr_ndcg(qrun, qrels, tgt_n=1000):
    """
    Compute the per-ranking nDCG metric for Task 1.

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
    disc = discount(tgt_n)

    # compute nDCG
    run_dcg = np.sum(util * disc[:n])
    ideal_dcg = np.sum(disc[:rel_n])

    return run_dcg / ideal_dcg


def qr_awrf(qrun, page_align, qtgt):
    """
    Compute the per-ranking AWRF metric for Task 1.

    Args:
        qrun(array-like):
            The (ranked) document identifiers returned for a single query.
        page_align(pandas.DataFrame):
            A Pandas data frame whose index is page IDs and columns are page alignment
            values for each fairness category.
        qtgt(array-like):
            The target distribution for this query.
    Returns:
        float:
            The AWRF score for the ranking.
    """

    # length of run
    n = len(qrun)

    disc = discount(n)

    # get the page alignments
    #ralign = page_align.loc[qrun]
    ralign = page_align.reindex(qrun)
    # discount and compute the weighted sum
    ralign = ralign.multiply(disc, axis=0)
    ralign = ralign.sum(axis=0) / np.sum(disc)
    if np.sum(ralign) == 0:
        ralign = np.ones_like(ralign)
    # now we have an alignment vector
    dist = jensenshannon(ralign, qtgt)
    # JS distance is sqrt of divergence
    return 1 - dist * dist


class Task1Metric:
    """
    Task 1 metric implementation.

    The metric is defined as the product of the nDCG and the AWRF.  This class stores the
    data structures needed to look up qrels and target alignments for each query.  It is
    callable, and usable directly as the function in a Pandas :meth:`pandas.DataFrameGroupBy.apply`
    call on a data frame that has been grouped by query identifier::

        run.groupby('qid')['page_id'].apply(metric)

    Since :meth:`__call__` extracts query ID directly from the name, this doesn't work if you
    have a frame that you are grouping by more than one column.
    """

    def __init__(self, qrels, page_align, qtgts):
        """
        Construct Task 1 metric.

        Args:
            qrels(pandas.DataFrame):
                The data frame of relevant documents, indexed by query ID.
            page_align(pandas.DataFrame):
                The data frame of page alignments for fairness criteria, indexed by page ID.
            qtgts(pandas.DataFrame):
                The data frame of query target distributions, indexed by query ID.
        """
        self.qrels = qrels
        self.page_align = page_align
        self.qtgts = qtgts
    
    def __call__(self, run):
        if isinstance(run.name, tuple):
            qid = run.name[-1]
        else:
            qid = run.name
        qrel = self.qrels.loc[qid]
        qtgt = self.qtgts.loc[qid]

        ndcg = qr_ndcg(run, qrel, 1000)
        assert ndcg >= 0
        assert ndcg <= 1
        awrf = qr_awrf(run, self.page_align, qtgt)
        # assert awrf >= 0
        # assert awrf <= 1

        return pd.Series({
            'nDCG': ndcg,
            'AWRF': awrf,
            'Score': ndcg * awrf
        })


def qr_exposure(qrun, page_align):
    """
    Compute the group exposure from a single ranking.

    Args:
        qrun(array-like):
            The (ranked) document identifiers returned for a single query.
        page_align(pandas.DataFrame):
            A Pandas data frame whose index is page IDs and columns are page alignment
            values for each fairness category.
    Returns:
        numpy.ndarray:
            The group exposures for the ranking (unnormalized)
    """

    # length of run
    n = len(qrun)
    disc = discount(n)

    # get the page alignments
    ralign = page_align.reindex(qrun)
    # discount and compute the weighted sum
    ralign = ralign.multiply(disc, axis=0)
    ralign = ralign.sum(axis=0)

    assert len(ralign) == page_align.shape[1]

    return ralign


def qrs_exposure(qruns, page_align):
    """
    Compute the group exposure from a sequence of rankings.

    Args:
        qruns(array-like):
            A data frame of document identifiers for a single query.
        page_align(pandas.DataFrame):
            A Pandas data frame whose index is page IDs and columns are page alignment
            values for each fairness category.
    Returns:
        pandas.Series:
            Each group's expected exposure for the ranking.
    """

    rexp = qruns.groupby('seq_no')['page_id'].apply(qr_exposure, page_align=page_align)
    exp = rexp.unstack().fillna(0).mean(axis=0)
    assert len(exp) == page_align.shape[1]

    return exp


def qw_tgt_exposure(qw_counts: pd.Series) -> pd.Series:
    """
    Compute the target exposure for each work level for a query.

    Args:
        qw_counts(pandas.Series):
            The number of articles the query has for each work level.
    
    Returns:
        pandas.Series:
            The per-article target exposure for each work level.
    """
    if 'id' == qw_counts.index.names[0]:
        qw_counts = qw_counts.reset_index(level='id', drop=True)
    qwc = qw_counts.reindex(work_order, fill_value=0).astype('i4')
    tot = int(qwc.sum())
    da = discount(tot)
    qwp = qwc.shift(1, fill_value=0)
    qwc_s = qwc.cumsum()
    qwp_s = qwp.cumsum()
    res = pd.Series(
        [np.mean(da[s:e]) for (s, e) in zip(qwp_s, qwc_s)],
        index=qwc.index
    )
    return res


class Task2Metric:
    """
    Task 2 metric implementation.

    This class stores the data structures needed to look up qrels and target exposures for each
    query.  It is callable, and usable directly as the function in a Pandas
    :meth:`pandas.DataFrameGroupBy.apply` call on a data frame that has been grouped by query
    identifier::

        run.groupby('qid').apply(metric)

    Since :meth:`__call__` extracts query ID directly from the name, this doesn't work if you have a
    frame that you are grouping by more than one column.
    """

    def __init__(self, qrels, page_align, page_work, qtgts):
        """
        Construct Task 2 metric.

        Args:
            qrels(pandas.DataFrame):
                The data frame of relevant documents, indexed by query ID.
            page_align(pandas.DataFrame):
                The data frame of page alignments for fairness criteria, indexed by page ID.
            qtgts(pandas.DataFrame):
                The data frame of query target exposures, indexed by query ID.
        """
        self.qrels = qrels
        self.page_align = page_align
        self.page_work = page_work
        self.qtgts = qtgts
    
    def __call__(self, sequence):
        if isinstance(sequence.name, tuple):
            qid = sequence.name[-1]
        else:
            qid = sequence.name
        qtgt = self.qtgts.loc[qid]

        s_exp = qrs_exposure(sequence, self.page_align)
        avail_exp = np.sum(discount(50))
        tgt_exp = qtgt * avail_exp
        delta = s_exp - tgt_exp

        ee_disp = np.dot(s_exp, s_exp)
        ee_rel = np.dot(s_exp, tgt_exp)
        ee_loss = np.dot(delta, delta)

        return pd.Series({
            'EE-L': ee_loss,
            'EE-D': ee_disp,
            'EE-R': ee_rel,
        })