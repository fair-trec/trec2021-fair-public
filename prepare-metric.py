"""
Prepare a metric for TREC Fair Ranking.

The TREC Fair Ranking 2021 metrics require data for proper computation - qrels, alignments,
etc.  This script prepares that data.  It reads the relevant data from the 'data' directory
and pickles an instance of the appropriate metric class (from 'metrics.py') that has this
data embedded in it.  Code wanting to evaluate runs can unpickle the metric and apply it
to their runs.  By default, this file is gzip-compressed.

Usage:
    prepare-metric.py [options] (--task1 | --task2)

Options:
    --task1
        Compile the Task 1 metric.
    --task2
        Compile the Task 2 metric.
    --topics FILE
        Read topics and qrels from FILE [default: data/trec_topics.json.gz].
    -o FILE, --output FILE
        Pickle the metric to FILE.
    --verbose
        Increase logging verbosity.
"""

import sys
from pathlib import Path
import logging
from docopt import docopt
import pickle
import gzip

import pandas as pd
import numpy as np

import metrics

_log = logging.getLogger('oracle-runs')

def main(opts):
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

    dir = Path('data')
    meta_file = dir / 'trec_metadata.json.gz'
    topic_file = Path(opts['--topics'])

    if opts['--task1']:
        metric = metrics.Task1Metric.load(meta_file, topic_file)
        dft_out = 'Task1Metric.pkl.gz'
    elif opts['--task2']:
        metric = metrics.Task2Metric.load(meta_file, topic_file)
        dft_out = 'Task2Metric.pkl.gz'
    
    out_file = opts['--output']
    if out_file is None:
        out_file = dft_out
    _log.info('writing metric to %s', out_file)
    with gzip.open(out_file, 'wb') as f:
        pickle.dump(metric, f)


if __name__ == '__main__':
    opts = docopt(__doc__)
    main(opts)