"""
Produce TREC fair ranking runs from an oracle.

This script assumes the data lives in a directory 'data'.  It loads the training
topics and uses them as an oracle for producing rankings.

Usage:
    oracle-runs.py --task1 [options]
    oracle-runs.py --task2 [options]

Options:
    -v, --verbose
        Write verbose logging output.
    -o FILE
        Write output to FILE.
    -p PREC, --precision=PREC
        Produce results with the specified precision [default: 0.9].
"""

import sys
from pathlib import Path
import logging
from tqdm import tqdm
from docopt import docopt

import pandas as pd
import numpy as np

_log = logging.getLogger('oracle-runs')


def load_metadata():
    meta_f = Path('data/trec_metadata.json.gz')
    _log.info('reading %s', meta_f)
    meta = pd.read_json(meta_f, lines=True, compression='gzip')
    return meta.set_index('page_id')


def load_topics():
    topic_f = Path('data/trec_topics.json.gz')
    _log.info('reading %s', topic_f)
    topics = pd.read_json(topic_f, lines=True, compression='gzip')
    return topics


def sample_docs(rng, meta, rel, n, prec):
    _log.debug('sampling %d rel items (n=%d, prec=%.2f)', len(rel), n, prec)
    n_rel = min(int(n * prec), len(rel))
    n_unrel = n - n_rel
    
    rel = np.array(rel)
    all = pd.Series(meta.index)
    unrel = all[~all.isin(rel)].values

    samp_rel = rng.choice(rel, n_rel, replace=False)
    samp_unrel = rng.choice(unrel, n_unrel, replace=False)
    samp = np.concatenate([samp_rel, samp_unrel])
    rng.shuffle(samp)
    return pd.Series(samp)


def task1_run(opts, meta, topics):
    rng = np.random.default_rng()
    rank_len = 1000
    prec = float(opts['--precision'])

    rels = topics[['id', 'rel_docs']].set_index('id').explode('rel_docs')

    def sample(df):
        return sample_docs(rng, meta, df['rel_docs'], rank_len, prec)
    
    runs = rels.groupby('id').apply(sample)
    runs.columns.name = 'rank'
    runs = runs.stack().reset_index(name='page_id')
    _log.info('sample runs:\n%s', runs)
    return runs[['id', 'page_id']]


def task2_run(opts, meta, topics):
    rng = np.random.default_rng()
    rank_len = 50
    run_count = 100
    prec = float(opts['--precision'])

    rels = topics[['id', 'rel_docs']].set_index('id').explode('rel_docs')

    def one_sample(df):
        return sample_docs(rng, meta, df['rel_docs'], rank_len, prec)
    
    def multi_sample(df):
        runs = dict((i+1, one_sample(df)) for i in tqdm(range(run_count), 'reps', leave=False))
        rdf = pd.DataFrame(runs)
        rdf.columns.name = 'rep_number'
        rdf.index.name = 'rank'
        return rdf.T
    
    runs = rels.groupby('id').progress_apply(multi_sample)
    runs = runs.stack().reset_index(name='page_id')
    _log.info('multi-sample runs:\n%s', runs)
    return runs[['id', 'rep_number', 'page_id']]


def main(opts):
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)
    tqdm.pandas()

    meta = load_metadata()
    topics = load_topics()

    if opts['--task1']:
        runs = task1_run(opts, meta, topics)
        dft_out = 'task1.tsv'
    elif opts['--task2']:
        runs = task2_run(opts, meta, topics)
        dft_out = 'task2.tsv'
    else:
        raise ValueError('no task specified')
    
    out_file = opts.get('-o', dft_out)
    _log.info('writing to %s', out_file)
    runs.to_csv(out_file, index=False)


if __name__ == '__main__':
    opts = docopt(__doc__)
    main(opts)