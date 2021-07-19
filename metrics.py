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
    ralign = page_align.loc[qrun]
    # discount and compute the weighted sum
    ralign = ralign.multiply(disc, axis=0)
    ralign = ralign.sum(axis=0) / np.sum(disc)
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

        run.groupby('qid')['doc_id'].apply(metric)

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
        qid = run.name
        qrel = self.qrels.loc[qid]
        qtgt = self.qtgts.loc[qid]

        ndcg = qr_ndcg(run, qrel, 1000)
        assert ndcg >= 0
        assert ndcg <= 1
        awrf = qr_awrf(run, self.page_align, qtgt)
        assert awrf >= 0
        assert awrf <= 1

        return pd.Series({
            'nDCG': ndcg,
            'AWRF': awrf,
            'Score': ndcg * awrf
        })

    @classmethod
    def load(cls, meta_file, topic_file):
        _log.info('reading page metadata from %s', meta_file)
        pages = pd.read_json(meta_file, lines=True)
        page_exp = pages[['page_id', 'geographic_locations']].explode('geographic_locations')

        _log.info('computing page alignments')
        page_align = page_exp.assign(x=1).pivot(index='page_id', columns='geographic_locations', values='x')
        page_align = page_align.iloc[:, 1:]
        page_align.fillna(0, inplace=True)

        _log.info('reading topics from %s', topic_file)
        topics = pd.read_json(topic_file, lines=True)
        qrels = topics[['id', 'rel_docs']].explode('rel_docs').reset_index(drop=True)

        _log.info('computing query alignments')
        qalign = qrels.join(page_align, on='rel_docs').groupby('id').sum()
        # normalize query alignments
        qa_sums = qalign.sum(axis=1)
        qalign = qalign.divide(qa_sums, axis=0)

        qtarget = (qalign + world_pop) * 0.5

        return cls(qrels.set_index('id'), page_align, qtarget)


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
    ralign = page_align.loc[qrun]
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

    rexp = qruns.groupby('rep_number')['page_id'].apply(qr_exposure, page_align=page_align)
    exp = rexp.unstack().fillna(0).mean(axis=0)
    assert len(exp) == page_align.shape[1]

    return exp


class Task2Metric:
    """
    Task 2 metric implementation.

    This class stores the data structures needed to look up qrels and target exposures for each
    query.  It is callable, and usable directly as the function in a Pandas
    :meth:`pandas.DataFrameGroupBy.apply` call on a data frame that has been grouped by query
    identifier::

        run.groupby('qid')['doc_id'].apply(metric)

    Since :meth:`__call__` extracts query ID directly from the name, this doesn't work if you have a
    frame that you are grouping by more than one column.
    """

    def __init__(self, qrels, page_align, qtgts):
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
        self.qtgts = qtgts
    
    def __call__(self, sequence):
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

    @classmethod
    def load(cls, meta_file, topic_file):
        _log.info('reading page metadata from %s', meta_file)
        pages = pd.read_json(meta_file, lines=True)
        page_exp = pages[['page_id', 'geographic_locations']].explode('geographic_locations')

        _log.info('computing page alignments')
        page_align = page_exp.assign(x=1).pivot(index='page_id', columns='geographic_locations', values='x')
        page_align.rename(columns={np.nan: 'Unknown'}, inplace=True)
        page_align.fillna(0, inplace=True)

        _log.info('reading topics from %s', topic_file)
        topics = pd.read_json(topic_file, lines=True)
        qrels = topics[['id', 'rel_docs']].explode('rel_docs').reset_index(drop=True)

        _log.info('computing query alignments')
        qalign = qrels.join(page_align, on='rel_docs').groupby('id').sum()

        # now some things get tricky. we need to average the *known* items with the
        # world population. so let's start by normalizing, then denormalizing
        qa_known = qalign.drop(columns=['Unknown'])
        # avoid divide-by-zero
        qa_ksums = qa_known.sum(axis=1)
        qa_kd = qa_known.divide(np.maximum(qa_ksums, 1.0e-6), axis=0)
        qa_kd = (qa_kd + world_pop) * 0.5
        # if something has *no* known-alignment things, it will require no
        # alignments. That seems unlikely!
        qa_known = qa_kd.multiply(qa_ksums, axis=0)
        
        qtarget = qalign[['Unknown']].join(qa_known)
        # and finally *actually* normalize the target distributions
        qtarget = qtarget.divide(qtarget.sum(axis=1), axis=0)

        return cls(qrels.set_index('id'), page_align, qtarget)