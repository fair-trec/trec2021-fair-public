# Fair TREC 2021 Public Code

This code is example implementations of the TREC 2021 metrics.

The `environment.yml` file defines a Conda environment that contains all required
dependencies.

## Oracle Runs

The `oracle-runs.py` file generates oracle runs from the training queries, with
a specified precision.  For example, to generate runs for Task 2 with precision
0.9, run:

    python .\oracle-runs.py -p 0.9 --task2 -o runs/task2-prec09.csv

These are useful for seeing the output format, and for testing metrics.

## Metrics

The metrics are defined in `metrics.py`.  The metrics require data (the qrels and
page alignments), and then are applied to data using Pandas data structures.

The `prepare-metric.py` script prepares a metric for use, pickling the usable
metric object.

The `Task1Metrics.py` and `Task2Metrics.py` scripts, and corresponding `.ipynb` files,
demonstrate how to load a prepared metric and evaluate a run.

## License

All code is licensed under the MIT License.