# Fair TREC 2021 Public Code

This code is example implementations of the TREC 2021 metrics.  To make it work,
you need to put data in `data`, as downloaded from TREC, and the runs in `runs`.

The `environment.yml` file defines a Conda environment that contains all required
dependencies.  It is **strongly recommended** to use this environment for running
the metric code.

If you store your runs in `runs` with a `.gz` extension, they will be found by
the evaluation code to run your own evaluations.

**Note:** the final evaluation metrics use gender data.  See the overview paper for
crucial notes on the limitations, warnings, and limitations, and ethical considerations
of this data.

Save the data files from [Fair TREC](https://fair-trec.github.io) into the `data`
directory before running this code.

## Oracle Runs

The `oracle-runs.py` file generates oracle runs from the training queries, with
a specified precision.  For example, to generate runs for Task 2 with precision
0.9, run:

    python .\oracle-runs.py -p 0.9 --task2 -o runs/task2-prec09.tsv

These are useful for seeing the output format, and for testing metrics.

## Metrics and Evaluation

The metrics are defined in `metrics.py`.  The metrics require data (the qrels and
page alignments), and then are applied to data using Pandas data structures.

The `Alignment.ipynb` notebook loads the data, computes alignments, and prepares
the metrics with their data for subsequent use.

The `Task1Evaluation.ipynb` and `Task2Evaluation.ipynb` notebooks use the prepared 
metrics to evaluate the runs.

Each of these notebooks is set up with Jupytext to pair with a Python script, so
you can just run the `.py` file if all you care about is the textual output. If
you use the Conda environment, Jupytext should automatically be set up so you can
use the notebooks and modify them without breaking the code/notebook link.

## License

All code is licensed under the MIT License.